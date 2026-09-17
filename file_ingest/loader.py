"""
loader.py —— 文件解析抽象与各格式实现

契约（见 file_achieve.md §3.2）：
- DocChunk：content + metadata
- Loader(ABC).load(path) -> list[DocChunk]
- 切块策略：
    .pdf  : pdfplumber，按页切块
    .pptx : python-pptx，每页文字+备注一个 chunk
    .docx : python-docx，按段落/标题层级切块
    .py   : tree-sitter，按 function/class 定义切块，metadata 带 name+lineno
    .ts/.go/.java : tree-sitter，同上
    其他代码文件   : 正则 fallback，按 def/class/func 行切块
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DocChunk:
    content: str
    metadata: dict = field(default_factory=dict)


class Loader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[DocChunk]:
        ...

    def _md(self, path: str, **extra) -> dict:
        p = Path(path)
        return {"source": str(p), "file_name": p.name, "file_type": p.suffix.lower().lstrip("."), **extra}


class PDFLoader(Loader):
    """pdfplumber：按页切块，每页一个 chunk。"""

    def load(self, path: str) -> list[DocChunk]:
        import pdfplumber  # 延迟导入：未用到不报错

        chunks: list[DocChunk] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(DocChunk(
                        content=text.strip(),
                        metadata=self._md(path, page=i, chunk_type="page"),
                    ))
        # 空文件兜底（避免返回 [] 被上层误判为解析失败）
        if not chunks:
            chunks.append(DocChunk(content="", metadata=self._md(path, page=0, chunk_type="page")))
        return chunks


class PPTXLoader(Loader):
    """python-pptx：每页文字 + 备注为一个 chunk。"""

    def load(self, path: str) -> list[DocChunk]:
        from pptx import Presentation

        prs = Presentation(path)
        chunks: list[DocChunk] = []
        for i, slide in enumerate(prs.slides, start=1):
            parts: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
            # 备注
            try:
                notes = slide.notes_slide.notes_text_frame.text
                if notes:
                    parts.append(f"[备注] {notes}")
            except Exception:
                pass
            content = "\n".join(parts).strip()
            if content:
                chunks.append(DocChunk(
                    content=content,
                    metadata=self._md(path, page=i, chunk_type="slide"),
                ))
        if not chunks:
            chunks.append(DocChunk(content="", metadata=self._md(path, page=0, chunk_type="slide")))
        return chunks


class DOCXLoader(Loader):
    """python-docx：按段落/标题层级切块（连续段落合并为一个语义块）。"""

    def load(self, path: str) -> list[DocChunk]:
        from docx import Document

        doc = Document(path)
        chunks: list[DocChunk] = []
        buf: list[str] = []
        heading: Optional[str] = None

        def flush():
            nonlocal buf, heading
            text = "\n".join(buf).strip()
            if text:
                chunks.append(DocChunk(
                    content=text,
                    metadata=self._md(path, heading=heading or "", chunk_type="paragraph"),
                ))
            buf = []
            heading = None

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style and para.style.name.startswith("Heading"):
                flush()
                heading = text
            else:
                buf.append(text)
        flush()

        if not chunks:
            chunks.append(DocChunk(content="", metadata=self._md(path, chunk_type="paragraph")))
        return chunks


# ---------------------------------------------------------------------------
# 代码切块：优先 tree-sitter，失败回退正则
# ---------------------------------------------------------------------------

# 各语言用于正则 fallback 的"定义起点"关键字
_CODE_DEF_PATTERNS = {
    ".py": re.compile(r"^\s*(def |class |async def )", re.MULTILINE),
    ".go": re.compile(r"^\s*(func |type |var |const )", re.MULTILINE),
    ".ts": re.compile(r"^\s*(function |class |export |interface |type |const |let |var )", re.MULTILINE),
    ".java": re.compile(r"^\s*(public |private |protected |class |interface |enum |void |static )", re.MULTILINE),
    ".js": re.compile(r"^\s*(function |class |export |const |let |var )", re.MULTILINE),
}

# 通用正则：从定义行里抽出名字（name）
_NAME_RE = re.compile(r"(?:def|class|func|function|interface|type|enum|void|static|public|private|protected|export|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _extract_name(line: str) -> str:
    m = _NAME_RE.search(line)
    return m.group(1) if m else ""


class CodeRegexLoader(Loader):
    """正则 fallback：按 def/class/func 行切块，metadata 带 name + lineno。"""

    def __init__(self, ext: str):
        self.pattern = _CODE_DEF_PATTERNS.get(ext, _CODE_DEF_PATTERNS[".py"])

    def load(self, path: str) -> list[DocChunk]:
        text = Path(path).read_text(errors="ignore")
        lines = text.splitlines()
        chunks: list[DocChunk] = []
        start = 0
        for i, line in enumerate(lines):
            if self.pattern.match(line):
                # 上一个块 [start, i-1]
                if i > start:
                    chunks.append(self._make(path, lines, start, i - 1))
                start = i
        if len(lines) > start:
            chunks.append(self._make(path, lines, start, len(lines) - 1))

        if not chunks:
            chunks.append(DocChunk(content=text, metadata=self._md(path, name="", lineno=0, chunk_type="file")))
        return chunks

    def _make(self, path: str, lines: list[str], a: int, b: int) -> DocChunk:
        content = "\n".join(lines[a : b + 1]).strip()
        return DocChunk(
            content=content,
            metadata=self._md(path, name=_extract_name(lines[a]), lineno=a + 1, chunk_type="symbol"),
        )


class TreeSitterLoader(Loader):
    """tree-sitter：按 function_definition / class_definition 切块。

    若 tree-sitter 或对应语言 grammar 不可用，自动降级为 CodeRegexLoader。
    """

    def __init__(self, language: str, ext: str):
        self.language = language
        self.ext = ext
        self._regex_fallback = CodeRegexLoader(ext)
        try:
            import tree_sitter  # noqa
            from tree_sitter_languages import get_parser
            self._parser = get_parser(language)
            self._available = True
        except Exception:
            self._available = False

    def load(self, path: str) -> list[DocChunk]:
        if not self._available:
            return self._regex_fallback.load(path)

        import tree_sitter
        text = Path(path).read_bytes()
        tree = self._parser.parse(text)
        root = tree.root_node

        # 递归收集 function_definition / class_definition 节点
        nodes: list[tree_sitter.Node] = []
        stack = [root]
        target = {"function_definition", "class_definition",
                  "method_declaration", "constructor_declaration"}
        while stack:
            node = stack.pop()
            if node.type in target:
                nodes.append(node)
            else:
                stack.extend(node.children)

        # 按字节区间切分，避免重叠：合并嵌套（方法属于类，只保留类级外层）
        nodes.sort(key=lambda n: (n.start_byte, -(n.end_byte - n.start_byte)))
        chunks: list[DocChunk] = []
        covered: list[tuple[int, int]] = []
        for n in nodes:
            a, b = n.start_byte, n.end_byte
            if any(a < ca and b > cb for ca, cb in covered):
                continue  # 被更大的外层（class）覆盖，跳过内层 method
            covered.append((a, b))
            content = text[a:b].decode("utf-8", errors="ignore").strip()
            name = ""
            for child in n.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8", errors="ignore")
                    break
            chunks.append(DocChunk(
                content=content,
                metadata=self._md(path, name=name, lineno=n.start_point[0] + 1, chunk_type="symbol"),
            ))

        if not chunks:
            # 没识别到任何定义 → 整文件作为一个 chunk（脚本型代码）
            content = text.decode("utf-8", errors="ignore").strip()
            chunks.append(DocChunk(content=content, metadata=self._md(path, name="", lineno=0, chunk_type="file")))
        return chunks


# ---------------------------------------------------------------------------
# 工厂：按扩展名选 Loader
# ---------------------------------------------------------------------------

def _code_loader(ext: str) -> Loader:
    if ext in (".py",):
        return TreeSitterLoader("python", ext)
    if ext in (".ts", ".js", ".jsx", ".tsx"):
        return TreeSitterLoader("javascript", ext)
    if ext in (".go",):
        return TreeSitterLoader("go", ext)
    if ext in (".java",):
        return TreeSitterLoader("java", ext)
    return CodeRegexLoader(ext)


def get_loader(path: str) -> Loader:
    ext = Path(path).suffix.lower()
    mapping = {
        ".pdf": PDFLoader(),
        ".pptx": PPTXLoader(),
        ".docx": DOCXLoader(),
    }
    if ext in mapping:
        return mapping[ext]
    if ext in _CODE_DEF_PATTERNS:
        return _code_loader(ext)
    # 未知文件：按文本整文件读（不报错，交由上层限制类型）
    return TextLoader()


class TextLoader(Loader):
    """未知扩展名的兜底：整文件作为一个 chunk。"""

    def load(self, path: str) -> list[DocChunk]:
        text = Path(path).read_text(errors="ignore")
        return [DocChunk(content=text, metadata=self._md(path, chunk_type="file"))]
