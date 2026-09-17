"""
router.py —— 文件意图硬路由（入口检测）

契约（file_achieve.md §3.3）：
has_file_intent(message) -> bool
判定条件（满足任一即 True）：
- message 含 attachments 字段且非空
- message 含 file_id 引用
- message 文本匹配文件扩展名正则（.pdf/.pptx/.docx/.py 等）
"""
from __future__ import annotations

import re
from typing import Union

# 支持的扩展名（与 loader 对齐）。用「或」结构，避免嵌套字符类 [][...]。
_EXT_GROUP = (
    r"pdf|pptx|docx|doc|xlsx|xls|md|txt|py|ts|js|jsx|tsx|go|java|rs|"
    r"json|yaml|yml|toml|csv|log"
)
# ① 扩展名出现在文本中（作为文件名，而非普通句子）
_EXT_RE = re.compile(rf"\b[\w./\-]+\.({_EXT_GROUP})\b", re.IGNORECASE)
# ② 明确的文件操作动词 + 文件名
_FILE_ACTION_RE = re.compile(
    rf"(?:上传|打开|读取|解析|索引|检索|查一下|看一下|看下|分析|总结|总结一下|提取)"
    rf"[\s\S]*?\b[\w./\-]+\.({_EXT_GROUP})\b",
    re.IGNORECASE,
)
# ③ file_id 引用形如 fid_xxx / file_id=xxx / "file_id": "xxx"
_FILE_ID_RE = re.compile(r"\b(?:file_)?id\s*[:=]\s*[\"']?([0-9a-fA-F\-]{8,})[\"']?", re.IGNORECASE)


def has_file_intent(message: Union[str, dict]) -> bool:
    """
    判断消息是否涉及文件操作。

    Args:
        message: 字符串（纯文本）或 dict（含 role/content/attachments/file_id 等字段）

    Returns:
        bool
    """
    # 归一化为文本 + 附件信息
    if isinstance(message, dict):
        attachments = message.get("attachments") or []
        if attachments:
            return True
        # file_id 引用（字典字段形式）
        for field in ("file_id", "fileIds", "file_ids"):
            if message.get(field):
                return True
        text = message.get("content") or message.get("text") or ""
    else:
        text = str(message)

    text = text.strip()
    if not text:
        return False

    # 条件 ③：file_id 引用（文本形式）
    if _FILE_ID_RE.search(text):
        return True

    # 条件 ②：显式文件操作动词（带或不带文件名都算）
    if _FILE_ACTION_RE.search(text):
        return True
    # ②-b：仅有文件操作动词（句末/句中，未紧跟具体文件名）—— 也视为文件意图
    if re.search(
        r"(?:上传|打开|读取|解析|索引|检索|分析|总结|提取)\s*(?:一下|这个|该|该)?\s*(?:文件|文档|附件|表格|报告|论文)?\s*$",
        text,
    ):
        return True

    # 条件 ①：文本中出现支持的文件扩展名
    if _EXT_RE.search(text):
        return True

    return False
