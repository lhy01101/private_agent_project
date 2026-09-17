"""
prompt.py —— 文件列表注入 system prompt

契约（file_achieve.md §3.4）：
build_file_list_prompt(store, active_file_ids) -> str
格式："你有以下文件可用：\n- {name} ({type}): {summary}\n- ..."
"""
from __future__ import annotations

from .store import FileStore


def build_file_list_prompt(store: FileStore, active_file_ids: list[str]) -> str:
    """
    生成当前可用文件列表，注入 system prompt。

    说明：由于 file_id 对应的 file_name/summary 元数据存在 Chroma metadata 里，
    这里通过 store.collection.get(where={"file_id": ...}) 回填；inline 模式的文件
    若需展示，调用方应在 ingest 后把 {file_id: {name,type,summary}} 也维护一份
    （本模块通过 _REGISTRY 提供轻量维护，见下方）。

    Args:
        store: FileStore 实例（用于查询 indexed 文件的元数据）
        active_file_ids: 当前对话活跃的文件 file_id 列表

    Returns:
        可直接拼接到 system prompt 的段落；无文件时返回空串
    """
    if not active_file_ids:
        return ""

    lines: list[str] = ["你有以下文件可用："]
    for fid in active_file_ids:
        info = _resolve_file_info(store, fid)
        if info is None:
            lines.append(f"- {fid} (未知文件)")
        else:
            lines.append(f"- {info['name']} ({info['type']}): {info['summary']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 文件元信息登记簿（供 system prompt 回填用）
#  ingest_file 返回结果后，Agent 应调用 register_file() 登记
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, dict] = {}


def register_file(file_id: str, file_name: str, file_type: str, summary: str) -> None:
    _REGISTRY[file_id] = {"name": file_name, "type": file_type, "summary": summary}


def unregister_file(file_id: str) -> None:
    _REGISTRY.pop(file_id, None)


def _resolve_file_info(store: FileStore, file_id: str) -> dict | None:
    """优先 _REGISTRY（含 inline 文件），其次查 Chroma metadata。"""
    if file_id in _REGISTRY:
        return _REGISTRY[file_id]
    try:
        res = store.collection.get(
            where={"file_id": file_id},
            limit=1,
        )
        metas = res.get("metadatas") or []
        if not metas:
            return None
        md = metas[0]
        return {
            "name": md.get("file_name", file_id),
            "type": md.get("file_type", "unknown"),
            "summary": md.get("summary", ""),
        }
    except Exception:
        return None
