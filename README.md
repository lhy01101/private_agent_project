# Agent Tool Routing & RAG

一个从零搭建的 Agent 项目：支持**工具调用、联网搜索、RAG 知识库检索、多轮记忆**，并围绕"工具过多容易选错"这一核心痛点，设计并实现了一套**基于语义距离的工具路由层**。

---

## 为什么写这个项目

大语言模型通过 function calling 使用工具，但当工具数量变多、描述相似时，**模型经常选错工具**——这是所有 Agent 落地都会撞上的墙。

以前是靠优化提示词或者人工写 if/elif 分发，前者不稳定，后者可扩展性差。

让我们现代化一点，这里的思路是：**让 query 自己做路由**。把用户提问做一次 embedding，计算它与每个工具描述的语义距离，按分数自动选工具。这样新增工具只需补一条描述，调用方零改动。

---

## 功能特性

- [x] **工具语义路由**：基于 embedding 相似度自动选择工具，替代手工 if/elif
- [x] **负例抑制机制**：打分从单一"正分"升级为 `正分 − 负分`，显著降低误绑率
- [x] **兜底 + 底线**：保证工具集永不为空，同时用 `fallback_min_net` 防止"什么都强制绑工具"
- [x] **网络搜索语义路由**：博查 / Tavily / DuckDuckGo 多供应商，按语种与可用性自动切换
- [x] **RAG 检索 + Cross-Encoder 重排序**：top50 粗排 → cross-encoder 精排回 top3
- [x] **增量索引**：`build_index.py` 只处理新增/变更文件，无需全量重建
- [x] **多轮记忆**：基于 checkpointer 的短期记忆
- [x] **流式输入输出**：Gradio 网页端实时对话
- [x] **动态模型选择中间件**：运行时按策略切换底层模型
- [ ] 跨对话长期记忆（主动写入本地文档）—— *进行中*
- [ ] Agent 文件读写能力（"不仅能读，还要能写"）—— *规划中*

---

## 项目结构

```
RAG/
├── wonder_agent.py            ★ Agent 装配入口（唯一 create_agent 的地方）
├── web_chat_box.py            ★ Gradio 网页入口（uv run python web_chat_box.py → :7860）
├── pyproject.toml / uv.lock     uv 管理依赖，Python 3.12
│
├── self_packages/             ★ 核心功能包
│   ├── tools.py                 工具池汇总（唯一注册点）
│   ├── dynamic_select.py        basic/advanced 模型 + DynamicModelMiddleware
│   ├── tool_routing_middleware.py  ToolRoutingMiddleware（按语义裁剪工具）
│   ├── tool_router.py           SemanticToolRouter（embedding 打分路由引擎）
│   ├── routes_archive.py        路由表 ROUTES（7 条路由的语料/负例/互斥/工具映射）
│   ├── prompts.py               【人格】+ 工具规则拼装 system prompt
│   ├── web_search.py            搜索聚合（博查/Tavily/DDG，按 CJK 分流）
│   ├── web_search_provider.py   三家搜索的后端实现
│   ├── update_index.py          增量建索引（mtime+size 指纹 → file_state.json）
│   ├── build_index.py           早期全量建索引脚本（已被 update_index 取代）
│   ├── chat_box.py              CLI 对话入口（保留，web 入口为主）
│   ├── permission.py            user_id → basic/pro/quant 权限表 + runtime context
│   ├── response_format.py       结构化输出 schema（当前因模型不适配被注释掉）
│   ├── calibrate.py             路由调参/评测脚本（内嵌测试语料）
│   ├── unitest_sample.py        示例单测
│   └── .env                     API keys
│
├── file_ingest/               ★ 新增模块：文件读取与检索（2026-09-16）
│   ├── __init__.py              对外 API + 惰性导入 langchain 依赖
│   ├── loader.py                多格式解析：pdf/pptx/docx/py/ts/go…
│   ├── store.py                 FileStore：Chroma 封装，按 file_id 隔离
│   ├── router.py                has_file_intent() 文件意图硬路由（正则）
│   ├── prompt.py                文件清单注入 system prompt
│   ├── tools.py                 ingest_file / query_file 两个 Tool
│   └── tests/                   8 个测试文件（含 live chroma、真实文档）
│
├── character/                   人设源文件：IDENTITY / SOUL / USER / Intimacy.md
├── chroma_rag/                  向量库数据（chroma.sqlite3 + file_state.json）
├── docs/                        RAG 语料（README/log/optimize.md 的副本）
└── pictures/                    头像等静态资源 
```

---

## 快速开始

### 1. 准备文档与索引

```bash
mkdir docs
# 把你的 md/txt/pdf 丢进 docs/
cp ~/notes/*.md docs/
cp ~/project/*.pdf docs/

python update_index.py
```

`update_index.py` 会扫描 `docs/` 所有文件，对比 `file_state.json`，**只处理新文件和 mtime/size 变了的文件**，切片后追加进已有 Chroma。

> ⚠️ 若更改了 `chunk_size` 或 `chunk_overlap`，需删除旧库重建，否则切片大小不一致会导致检索精度下降：
> ```bash
> rm -r chroma_rag
> ```

### 2. 配置环境变量

在 `.env` 中填入所需 Key（API Key 不要硬编码进代码）：

```env
DEEPSEEK_API_KEY=xxx     # 可以去self_packages/dynamic_select.py把basic_model与advanced_model换成你想要的模型
BOCHA_API_KEY=xxx        # 博查（中文搜索）
TAVILY_API_KEY=xxx       # Tavily（英文搜索）
HF_TOKEN=xxx             # HuggingFace（cross-encoder / embedding）
HF_HUB_OFFLINE=1         # 在下载好embedding模型后可以设置离线模式以提速
DEV_DDG_FALLBACK=1       # 开发环境启用 DuckDuckGo 兜底
```

### 3. 启动

```bash
python web_chat_box.py            # Gradio 网页端
```

---

## 核心设计：工具语义路由

这是本项目最有价值的部分，也是我花时间最多的地方。

### 问题

工具多了以后，模型容易"乱选"。比如"看一下时间"这种短句，可能同时命中 `web_search`、`rag`、`timezone` 三个工具的描述，导致一次简单提问触发一堆无关调用。

### 方案：两层打分 + 负例抑制

不是把 query 分配到不同 Agent 管线，而是**对 query 做一次 embedding，根据语义距离选择工具输入**。

打分逻辑演进过程：

```
v1: score = 0.7 × example_sim + 0.3 × desc_sim        # 只看正例，容易误绑
v2: net_score = pos_score − 0.5 × neg_sim             # 引入负例抑制
```

| 项 | 含义 |
|---|---|
| `pos_score` | 正例相似度 + 工具描述相似度 |
| `neg_sim` | 与 `negative_examples` 的最大余弦相似度（越像负例越高） |
| `net_score` | 净分，被负例拉低 |

阈值校准通过 `calibrate.py` 完成：`threshold` 决定哪些工具入选，`fallback_min_net` 是兜底底线。

### 兜底机制与它的副作用

早期为了保证"工具集永不为空"（从而让模型始终走结构化 function calling，避免 DSML 退化），我加了赢家兜底。

但兜底有个代价：**几乎任何 query 都会强制绑一个工具**——"哈哈哈笑死"也会去调时间工具。

解决：在兜底中加入底线 `fallback_min_net = 0.30`：
- 净分太低 → 返回空，让模型直接回答（闲聊不再绑工具）
- 保留对 "AI 新闻"(0.522) 这类边界 query 的兜底救援

这是一个典型的 **trade-off**：宁可放过，不可错杀——但要给边界 case 留救援通道。

### 路由收敛问题（已知）

加入 `direct_answer` 路由后出现了"塞一堆工具"的现象，根源是两点：
1. `route()` 用 `>= threshold` 筛选，没有"只取最佳"的收敛
2. `direct_answer` 缺少"互斥优先权"，导致工具路由们一起过线

属于"选错工具但侥幸答对"一类，仍在优化。详见下方[踩坑记录](#问题记录)。

---

## 网络搜索路由

不引入额外 embedding 模型，靠**轻量启发式 + 配置态**决定，避免每次搜索多花一次 LLM 调用。

**配置层（硬路由）**——看哪个 Key 存在：
- `BOCHA_API_KEY` → 博查可用
- `TAVILY_API_KEY` → Tavily 可用
- `DEV_DDG_FALLBACK=1` → DDG 进候选

**查询层（软路由）**——对 query 做轻量判断：
- 含中日韩字符 → 优先博查（中文索引强）
- 纯 ASCII 且像英文短语/技术名 → 博查失败再走 Tavily
- 博查调用异常（429/超时/空结果）→ 自动跳下一个可用供应商

**容错链**：供应商按顺序 try，全部失败返回固定错误串，**不让 LangGraph ToolNode 抛异常炸图**。

> 注：博查的本地化搜索是基于请求 IP 在服务端完成的，AI 本身看不到你的 IP。

---

## RAG 设计

- **检索流程**：query → Chroma 取 top50 → cross-encoder 重排序 → 返回 top3
- **Embedding 选型**：通用 embedding（bge）专为"找文档"训练，区别于 LLM hidden state（专为"说人话"训练，检索弱）。后续计划迁移到 `bge-m3` 以更好支持中英混合 query。
- **切片策略**：调整 `chunk_size` / `chunk_overlap` 以适应碎片化文本；markdown 考虑结构化切分。

**语义路由示例：**

| 用户问题 | 路由到 |
|---|---|
| "怎么配置 nginx" | 技术文档知识库 |
| "年假还剩几天" | 人事制度知识库 |
| "帮我写周报" | 纯 LLM（不走 RAG） |
| "你是谁" | 固定回答（不走 LLM） |

即：**用向量相似度代替 if/elif 做请求分发。**

---

## 问题记录

- **DSML 退化**：模型在"手里没有可用工具"时会把工具调用写成正文文本。根因是 tool_router 参数与模型意愿相差过大，靠兜底保证工具集非空解决。
- **上下文记忆失效**：重新加入 `checkpointer=InMemorySaver()` 后恢复短期记忆；去除了会被误认为用户信息的 `get_user_location` 工具。
- **chunksize 不一致**：更改切片大小后未重建索引，导致检索出错、语义近似度精度降低——需删库重建。
- **"选错但答对"**：路由选错工具，但容错链兜底拿到了正确结果。比直接报错更值得查，因为掩盖了路由缺陷。
- **中间件与预绑定模型冲突**：中间件不能与已调用 `bind_tools` 的模型一起使用，参数需与工具参数绑定。
- **权限设计**：`permission.py` 用一张 `USER_PERMISSIONS` 表集中管理，后续接真实权限服务时只需把 dict 换成数据库查询，调用方零改动。

---

## 后续方向

- [ ] 迁移 embedding 到 `bge-m3`，根治中英混合 query 检索
- [ ] Agent 文件读写能力（不仅"读"，还能"写"文档）
- [ ] 跨对话长期记忆：按日期归档，主动写入本地文档（不依赖服务器运行）
- [ ] FastAPI 异步页面 + 文件输入 + 多轮对话
- [ ] 动态系统提示（根据用户输入选择提示词）
- [ ] 多步骤协作（AgentExecutor / LangGraph）
- [ ] AI 主动询问用户能力
- [ ] 主动输出算法：特定时段 / 情感需求期，基于近期记忆主动向用户发起聊天

---

## 环境依赖

详见 `pyproject.toml`。

---

## 许可证
MIT
This project is licensed under the MIT [License](LICENSE).