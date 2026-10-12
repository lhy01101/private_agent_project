"""文件读取工具：按路径读取本地文本文件（txt/md/json/py 等），对齐后返回纯文本给模型。"""

import os
from langchain.tools import tool

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php",
    ".html", ".htm", ".css", ".xml", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".csv", ".sh", ".bat", ".ps1",
    ".sql", ".r", ".swift", ".kt", ".scala", ".lua", ".zig",
}

MAX_CHARS = 60_000

# 工作区根目录（项目根）
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 扫描时跳过的目录
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "chroma_rag", "uploads", ".qoder-cn", "static",
}


def _align_text(raw: str) -> str:
    """对齐文本：统一换行、压缩连续空行、去除首尾空白。"""
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


@tool
def read_local_file(file_path: str) -> str:
    """读取本地文件内容。输入文件的绝对或相对路径，支持 txt/md/json/py 等文本文件。"""
    path = os.path.expanduser(file_path)
    if not os.path.isfile(path):
        return f"错误：文件不存在 - {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"错误：不支持的文件类型 - {ext}。支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                raw = f.read()
        except Exception as e:
            return f"错误：无法读取文件 - {e}"
    except Exception as e:
        return f"错误：无法读取文件 - {e}"

    aligned = _align_text(raw)
    if len(aligned) > MAX_CHARS:
        aligned = aligned[:MAX_CHARS] + f"\n\n... [截断：原文超过 {MAX_CHARS} 字符]"

    return f"文件：{path}\n类型：{ext}\n---\n{aligned}"


@tool
def list_workspace(subdir: str = "") -> str:
    """扫描工作区目录结构，返回文件树。可传入子目录路径（如 'self_packages'）只扫描该子目录。"""
    target = os.path.join(WORKSPACE_ROOT, subdir) if subdir else WORKSPACE_ROOT
    target = os.path.normpath(target)

    if not os.path.isdir(target):
        return f"错误：目录不存在 - {target}"

    # 安全检查：不允许扫描工作区之外的目录
    if not target.startswith(os.path.normpath(WORKSPACE_ROOT)):
        return "错误：不允许扫描工作区之外的目录"

    lines = []
    rel_base = os.path.relpath(target, WORKSPACE_ROOT)
    prefix = f"{rel_base}/" if rel_base != "." else ""

    for root, dirs, files in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        rel = os.path.relpath(root, WORKSPACE_ROOT)
        depth = rel.count(os.sep) if rel != "." else 0
        if depth > 3:
            dirs.clear()
            continue
        indent = "  " * depth
        dir_name = os.path.basename(root)
        lines.append(f"{indent}📁 {dir_name}/")
        for f in sorted(files):
            if f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS or ext in {".png", ".jpg", ".jpeg", ".gif", ".ico"}:
                lines.append(f"{indent}  📄 {f}")

    if not lines:
        return f"目录 {prefix or '/'} 下无可读文件"

    header = f"工作区：{WORKSPACE_ROOT}\n扫描范围：{prefix or '/'}\n---"
    return header + "\n" + "\n".join(lines)
