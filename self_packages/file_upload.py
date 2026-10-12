"""文件上传与读取：保存上传文件，按格式读取并对齐文本后返回给模型。"""

import os
import uuid

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php",
    ".html", ".htm", ".css", ".xml", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".csv", ".sh", ".bat", ".ps1",
    ".sql", ".r", ".swift", ".kt", ".scala", ".lua",
}

MAX_CHARS = 60_000


def is_supported(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def save_upload(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def align_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out = []
    blank = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(stripped)
    return "\n".join(out).strip()


def read_file_content(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            raw = f.read()

    aligned = align_text(raw)
    if len(aligned) > MAX_CHARS:
        aligned = aligned[:MAX_CHARS] + f"\n\n... [截断：原文超过 {MAX_CHARS} 字符]"
    return aligned


def format_file_for_message(filename: str, content: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return f"[文件: {filename} (类型: {ext})]\n---\n{content}\n---"
