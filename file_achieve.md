# 工程实现 Brief：文件管理模块

> 版本：v1.0  
> 日期：2026-09-15  
> 状态：可开工  
> 关联设计文档：FILE_MANAGEMENT_ARCHITECTURE.md

---

## 一、目标

为现有 Agent 系统增加文件读取与检索能力，支持：
- 用户上传文件（PDF / PPTX / DOCX / 代码源文件）
- 小文件全文直读、大文件切块后向量检索
- 文件以 `file_id` 形式被 Agent 管理，支持多轮对话中反复引用
- 与现有 Chroma 向量库、bge-m3 embedding、语义路由集成

---

## 二、不做的事（明确边界）

| 不做 | 原因 |
|---|---|
| 文件内容单独落盘存储 | metadata 记 path 即可溯源 |
| 跨用户文件共享 | 当前单用户，多用户隔离留接口不做实现 |
| 文件 OCR（扫描件） | 本期不支持，返回错误提示即可 |
| 图片/图表内容分析 | 本期不支持 |
| 文件在线预览/下载 | 前端能力，不在本模块范围 |
| 生命周期自动清理 | 接口预留，实现延后 |
| 查询翻译（中→英） | bge-m3 原生支持中英混合 |

---

## 三、接口定义

### 3.1 对外暴露的两个 Tool（注册到 Agent 工具池）

#### `ingest_file`

```python
@tool
def ingest_file(path: str, persist: bool = False) -> str:
    """
    解析并索引用户上传的文件。
    
    Args:
        path: 文件本地路径（绝对路径）
        persist: True=持久化保存（跨会话），False=临时（会话结束可清）
    
    Returns:
        JSON 字符串，包含：
        {
            "file_id": "uuid",
            "file_name": "report.pdf",
            "file_type": "pdf",
            "mode": "inline" | "indexed",
            "chunk_count": 0,        # inline 时为 0
            "summary": "...",        # 文件摘要（1-3句）
            "lifecycle": "temp" | "persistent"
        }
    """
```

**行为逻辑**：
1. 根据扩展名选择 Loader 解析文件
2. 估算 token 数（用 tiktoken 或字符数近似）
3. 若 ≤ 30,000 tokens → `mode=inline`，全文存入内存缓存（`_inline_cache[file_id] = text`），不写 Chroma
4. 若 > 30,000 tokens → `mode=indexed`，切块后写入 Chroma（metadata 带 `file_id`, `file_type`, `lifecycle`, `chunk_type`）
5. 生成文件摘要（前 500 字符 + 文件名 + 类型，拼接成一段描述）
6. 返回结果 JSON

#### `query_file`

```python
@tool
def query_file(file_id: str, question: str) -> str:
    """
    对已上传文件进行问答检索。
    
    Args:
        file_id: ingest_file 返回的 file_id
        question: 用户关于该文件的问题
    
    Returns:
        检索到的相关文本片段（拼接后的字符串），或全文（inline 模式）
    """
```

**行为逻辑**：
1. 查 `_inline_cache[file_id]` → 命中则直接返回全文
2. 未命中 → 用 bge-m3 对 `question` 做 embedding
3. Chroma 查询：`collection.query(embedding, where={"file_id": file_id}, n_results=5)`
4. 返回 Top-5 片段拼接（带页码/行号来源信息）

---

### 3.2 内部组件接口（工程师自行设计类结构，但需满足以下契约）

#### Loader 抽象

```python
@dataclass
class DocChunk:
    content: str
    metadata: dict  # {source, file_type, page/lineno, chunk_type, ...}

class Loader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[DocChunk]: ...
```

**各格式实现要求**：

| 扩展名 | Loader | 切块策略 |
|---|---|---|
| `.pdf` | pdfplumber | 按页切块，每页一个 chunk |
| `.pptx` | python-pptx | 每页文字 + 备注为一个 chunk |
| `.docx` | python-docx | 按段落/标题层级切块 |
| `.py` | tree-sitter | 按 function_definition / class_definition 切块，metadata 带 `name` + `lineno` |
| `.ts/.go/.java` | tree-sitter | 同上 |
| 其他代码文件 | 正则 fallback | 按 `def`/`class`/`func` 行切块 |

#### FileStore（Chroma 封装）

```python
class FileStore:
    def __init__(self, chroma_client, collection_name, embed_fn): ...
    def add(self, file_id: str, chunks: list[DocChunk], lifecycle: str) -> None: ...
    def query(self, file_id: str, query_embedding: list[float], k: int = 5) -> list[DocChunk]: ...
    def delete(self, file_id: str) -> None: ...
    def cleanup_temp(self) -> None: ...
    def set_embedder(self, embed_fn) -> None: ...
```

**要求**：
- 复用现有 Chroma 实例（同一个 `chroma_client`）
- `add` 时每个 chunk 的 metadata 必须包含 `file_id`
- `query` 时必须用 `where={"file_id": file_id}` 过滤

---

### 3.3 入口硬路由（文件检测）

```python
def has_file_intent(message: dict) -> bool:
    """
    判断消息是否涉及文件上传。
    
    判定条件（满足任一即返回 True）：
    - message 含 attachments 字段且非空
    - message 含 file_id 引用
    - message 文本匹配文件扩展名正则（.pdf/.pptx/.docx/.py 等）
    """
```

---

### 3.4 文件列表注入（system prompt 组装）

```python
def build_file_list_prompt(store: FileStore, active_file_ids: list[str]) -> str:
    """
    生成当前可用文件列表，注入 system prompt。
    
    输出格式：
    "你有以下文件可用：\n- {name} ({type}): {summary}\n- ..."
    """
```

---

## 四、数据流

### 4.1 上传文件流程

```
用户上传文件 → 系统收到消息（含 attachment path）
  ↓
has_file_intent() → True
  ↓
Agent 调 ingest_file(path, persist=False)
  ↓
Loader.load(path) → list[DocChunk]
  ↓
token 估算
  ├─ ≤ 30K → _inline_cache[file_id] = full_text
  └─ > 30K → FileStore.add(file_id, chunks, lifecycle="temp")
  ↓
返回 IngestResult JSON → Agent 记录到对话历史
```

### 4.2 问答流程

```
用户提问（可能涉及文件）
  ↓
语义路由 → 工具池含 query_file
  ↓
Agent 选 query_file(file_id, question)
  ↓
query_file 内部：
  ├─ inline → 返回缓存全文
  └─ indexed → embed(question) → Chroma where 过滤 → Top-5 → 拼接
  ↓
结果返回给 Agent → LLM 生成答案
```

---

## 五、验收标准（8 条）

| # | 验收项 | 验证方式 |
|---|---|---|
| 1 | 小文件（< 30K tokens）全文进缓存，Chroma 中 0 条记录 | 上传小文件后查 Chroma collection count |
| 2 | 大文件切块入库，metadata 含正确 `file_id` | 查 Chroma `where={"file_id": xxx}` 返回对应 chunk |
| 3 | `query_file` 对 inline 文件返回全文 | 上传小文件后立即 query，返回完整文本 |
| 4 | `query_file` 对 indexed 文件返回相关片段（非全文） | 上传大文件后问具体问题，返回 Top-5 片段 |
| 5 | `file_id` 隔离：query 只返回该文件的 chunk | 上传两个文件后分别 query，结果不交叉 |
| 6 | 代码文件切块带函数名 + 行号 metadata | 上传 .py 文件，查 chunk metadata 含 `name` + `lineno` |
| 7 | `ingest_file` 返回 JSON 含 file_id / mode / summary | 调用后解析返回字符串，字段齐全 |
| 8 | `has_file_intent` 对含扩展名的 query 返回 True | 单元测试：传入 "帮我看下 report.pdf" → True |

---

## 六、技术栈与依赖

| 组件 | 选型 | 版本要求 |
|---|---|---|
| 向量库 | ChromaDB | 复用现有 |
| Embedding | bge-m3 | 与现有知识库同一模型 |
| PDF 解析 | pdfplumber | latest |
| PPT 解析 | python-pptx | latest |
| DOCX 解析 | python-docx | latest |
| 代码解析 | tree-sitter + tree-sitter-languages | latest |
| Token 估算 | tiktoken（优先）或字符数近似 | - |
| Tool 框架 | LangChain @tool（或现有 Agent 框架的等价物） | 复用现有 |

---

## 七、目录结构建议

```
file_ingest/
├── __init__.py
├── loader.py        # Loader 抽象 + 各格式实现
├── chunker.py       # 切块策略（代码 AST / 文档语义）
├── store.py         # FileStore（Chroma 封装）
├── tools.py         # ingest_file / query_file @tool 定义
├── router.py        # has_file_intent 入口硬路由
├── prompt.py        # 文件列表注入 system prompt
└── tests/           # 验收测试
```

---

## 八、交付物

| 交付物 | 说明 |
|---|---|
| `file_ingest/` 模块代码 | 上述目录结构完整实现 |
| 验收测试脚本 | 8 条验收标准逐一验证 |
| 接口说明（本文档） | 已包含 |

---

## 九、联调要点

工程师完成后，需与以下现有模块联调：

| 现有模块 | 联调内容 | 负责人 |
|---|---|---|
| Agent Executor | 注册 `ingest_file` / `query_file` 到工具池 | 工程师 |
| 语义路由 | 工具池注入（含文件 Tool 时） | 工程师 |
| Chroma 实例 | 复用现有 client，不新建 | 工程师 |
| bge-m3 embedder | 复用现有 embedding 函数 | 工程师 |
| System prompt 组装 | 调用 `build_file_list_prompt` 注入 | 工程师 |

---

## 十、已知风险与应对

| 风险 | 应对 |
|---|---|
| pdfplumber 解析失败（扫描件/加密） | 捕获异常，返回"不支持此文件格式"错误 |
| tree-sitter 未安装对应语言 grammar | fallback 到正则切块 |
| Chroma where 过滤性能 | 当前数据量小，无需优化；量大时考虑单独 collection |
| 大文件 token 估算不准 | 用 tiktoken 精确计算，阈值可配置 |
| file_id 冲突 | 用 UUID4 生成 |

---

## 十一、后续迭代（本期不做，但接口预留）

- [ ] 生命周期自动清理（会话结束触发 `cleanup_temp`）
- [ ] 文件摘要用 small LLM 生成（当前用截断+拼接）
- [ ] 混合检索（BM25 + 向量）
- [ ] 多用户 file_id 归属校验
- [ ] rerank（bge-reranker-v2-m3）
- [ ] MCP 协议适配