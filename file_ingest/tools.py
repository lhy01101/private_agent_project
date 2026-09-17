"""
tools.py —— 对外暴露的两个 Tool：ingest_file / query_file

契约（file_achieve.md §3.1）：
- ingest_file(path, persist=False) -> JSON
    ≤30K tokens -> mode=inline，全文进 _inline_cache，不写 Chroma
    >30K tokens -> mode=indexed，切块写 Chroma（metadata 带 file_id/lifecycle/chunk_type）
- query_file(file_id, question) -> str
    inline -> 返回缓存全文
    indexed -> embed(question) -> Chroma where={"file_id": ...} -> Top-5 拼接
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from .loader import get_loader
from .store import FileStore


def _make_tool(func):
    """LangChain @tool 的惰性适配：有依赖就用 @tool，没有就当普通函数。

    这样 loader/store/router/prompt 完全不依赖 langchain，
    缺包时仍能跑单元测试与冒烟测试。
    """
    try:
        from langchain_core.tools import tool as _tool
        return _tool(func)
    except ImportError:
        func.invoke = lambda args: func(**args) if isinstance(args, dict) else func(args)
        return func


# ---------------------------------------------------------------------------
# 全局状态（单例）：由 Agent 启动时注入
# ---------------------------------------------------------------------------
_INLINE_CACHE: dict[str, str] = {}          # file_id -> 全文
_STORE: Optional[FileStore] = None          # 复用现有 Chroma 实例
_EMBED_FN: Optional[callable] = None        # bge-m3 embedder
_TOKEN_THRESHOLD = 30_000                   # 小文件阈值（tokens）


def configure(store: FileStore, embed_fn: callable) -> None:
    """Agent 启动时调用一次：注入 FileStore 与 bge-m3 embedder。"""
    global _STORE, _EMBED_FN
    _STORE = store
    _EMBED_FN = embed_fn
    store.set_embedder(embed_fn)


def _estimate_tokens(text: str) -> int:
    """token 估算：优先 tiktoken，失败则用字符数近似（≈ tokens * 4）。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4 + 1


def _make_summary(file_name: str, file_type: str, text: str) -> str:
    """文件摘要：文件名 + 类型 + 前 500 字符拼接。"""
    head = text.strip().replace("\n", " ")[:500]
    return f"{file_name}（{file_type}）：{head}"


def _new_file_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tool 1：ingest_file
# ---------------------------------------------------------------------------
@_make_tool
def ingest_file(path: str, persist: bool = False) -> str:
    """
    解析并索引用户上传的文件。

    Args:
        path: 文件本地绝对路径
        persist: True=持久化（跨会话，lifecycle=persistent），False=临时（lifecycle=temp）

    Returns:
        JSON 字符串：file_id / file_name / file_type / mode / chunk_count / summary / lifecycle
    """
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False)

    file_id = _new_file_id()
    file_name = p.name
    file_type = p.suffix.lower().lstrip(".") or "unknown"
    lifecycle = "persistent" if persist else "temp"

    try:
        loader = get_loader(str(p))
        chunks = loader.load(str(p))
    except Exception as e:
        return json.dumps({"error": f"不支持此文件格式或解析失败: {e}"}, ensure_ascii=False)

    # 全文拼接用于 token 估算与 inline 缓存
    full_text = "\n".join(c.content for c in chunks if c.content)
    token_count = _estimate_tokens(full_text)

    summary = _make_summary(file_name, file_type, full_text)

    if token_count <= _TOKEN_THRESHOLD:
        # 小文件：全文进缓存，不写 Chroma
        _INLINE_CACHE[file_id] = full_text
        mode = "inline"
        chunk_count = 0
    else:
        # 大文件：切块入库（必须有 store + embedder）
        if _STORE is None or _EMBED_FN is None:
            return json.dumps({"error": "FileStore 未初始化，请先调用 configure()"}, ensure_ascii=False)
        chunk_count = _STORE.add(file_id, chunks, lifecycle=lifecycle)
        mode = "indexed"

    result = {
        "file_id": file_id,
        "file_name": file_name,
        "file_type": file_type,
        "mode": mode,
        "chunk_count": chunk_count,
        "summary": summary,
        "lifecycle": lifecycle,
    }
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2：query_file
# ---------------------------------------------------------------------------
@_make_tool
def query_file(file_id: str, question: str) -> str:
    """
    对已上传文件进行问答检索。

    Args:
        file_id: ingest_file 返回的 file_id
        question: 用户关于该文件的问题

    Returns:
        检索到的相关文本片段拼接（带页码/行号来源），或全文（inline 模式）
    """
    # 1) inline 命中 -> 返回全文
    if file_id in _INLINE_CACHE:
        return _INLINE_CACHE[file_id]

    # 2) indexed -> embed + Chroma where 过滤
    if _STORE is None or _EMBED_FN is None:
        return json.dumps({"error": "FileStore 未初始化"}, ensure_ascii=False)

    try:
        q_emb = _EMBED_FN([question])[0]
    except Exception as e:
        return json.dumps({"error": f"embedding 失败: {e}"}, ensure_ascii=False)

    chunks = _STORE.query(file_id, q_emb, k=5)

    if not chunks:
        return json.dumps({"error": f"未找到 file_id={file_id} 的索引内容（可能已清理或尚未上传）"}, ensure_ascii=False)

    # 拼接，带来源信息（页码/行号）
    parts: list[str] = []
    for c in chunks:
        md = c.metadata
        src = ""
        if md.get("page"):
            src = f"[第{md['page']}页]"
        elif md.get("lineno"):
            src = f"[行{md['lineno']}{('/'+md['name']) if md.get('name') else ''}]"
        elif md.get("heading"):
            src = f"[§{md['heading']}]"
        score = md.get("score")
        score_str = f" (score={score:.3f})" if score is not None else ""
        parts.append(f"{src} {c.content}{score_str}".strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 仅供测试/调试：重置全局状态
# ---------------------------------------------------------------------------
def _reset_state() -> None:
    _INLINE_CACHE.clear()
