"""
file_ingest —— 文件读取与检索模块

对外 API（供 Agent 装配层使用）：
    from file_ingest import configure, ingest_file, query_file, has_file_intent, build_file_list_prompt

接入步骤（详见 file_achieve.md §九 联调要点）：
    1. configure(store, embed_fn)           # 启动时注入 Chroma + bge-m3
    2. has_file_intent(message)             # 入口硬路由（可选，用于前置分流）
    3. 注册 ingest_file / query_file 到 Agent 工具池
    4. build_file_list_prompt(store, ids)   # 注入 system prompt
"""
# tools 依赖 langchain_core，采用惰性导入：
#   from file_ingest import ingest_file, query_file, configure
# 若 langchain_core 未安装，首次访问时抛 ImportError（loader/store 等仍可用）
from .router import has_file_intent
from .prompt import build_file_list_prompt, register_file
from .store import FileStore
from .loader import get_loader, DocChunk, Loader


def __getattr__(name):
    if name in ("configure", "ingest_file", "query_file"):
        from . import tools
        return getattr(tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "configure",
    "ingest_file",
    "query_file",
    "has_file_intent",
    "build_file_list_prompt",
    "register_file",
    "FileStore",
    "get_loader",
    "DocChunk",
    "Loader",
]

__all__ = [
    "configure",
    "ingest_file",
    "query_file",
    "has_file_intent",
    "build_file_list_prompt",
    "register_file",
    "FileStore",
    "get_loader",
    "DocChunk",
    "Loader",
]
