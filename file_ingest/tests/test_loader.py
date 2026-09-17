"""
test_loader.py —— Loader 单元测试（不依赖 pdf/docx，聚焦切块逻辑与 metadata）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from file_ingest.loader import (
    get_loader,
    CodeRegexLoader,
    TextLoader,
    _CODE_DEF_PATTERNS,
)


class TestCodeRegexLoader:
    def test_py_splits_by_def_and_class(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text(
            'def foo():\n    return 1\n\n\nclass Bar:\n    """doc"""\n\n    def method(self):\n        pass\n'
        )
        chunks = CodeRegexLoader(".py").load(str(f))

        # 至少拆成 2 块（foo / Bar）
        assert len(chunks) >= 2
        names = {c.metadata.get("name") for c in chunks}
        assert "foo" in names
        assert "Bar" in names

        # 每个块都有 lineno
        assert all("lineno" in c.metadata for c in chunks)

    def test_lineno_correct(self, tmp_path):
        f = tmp_path / "b.py"
        content = '"""mod"""\n\n\ndef hello():\n    pass\n'
        f.write_text(content)
        chunks = CodeRegexLoader(".py").load(str(f))
        hello = next(c for c in chunks if c.metadata.get("name") == "hello")
        # 动态计算期望行号（不硬编码，避免受源码排版影响）
        expected = next(
            i + 1 for i, line in enumerate(content.splitlines())
            if line.startswith("def hello")
        )
        assert hello.metadata["lineno"] == expected

    @pytest.mark.parametrize("ext", list(_CODE_DEF_PATTERNS.keys()))
    def test_all_supported_extensions_load(self, tmp_path, ext):
        f = tmp_path / f"c{ext}"
        f.write_text(f"# test {ext}\n\nfunc main() {{}}\n\nclass A {{}}")
        chunks = CodeRegexLoader(ext).load(str(f))
        assert chunks  # 不抛异常且非空

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        chunks = CodeRegexLoader(".py").load(str(f))
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_type"] == "file"


class TestGetLoader:
    def test_unknown_ext_falls_back_to_text(self, tmp_path):
        f = tmp_path / "weird.xyz"
        f.write_text("hello")
        loader = get_loader(str(f))
        assert isinstance(loader, TextLoader)

    def test_returns_loader_instance(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x=1")
        assert get_loader(str(f)) is not None


class TestDocChunk:
    def test_metadata_default(self):
        from file_ingest.loader import DocChunk
        c = DocChunk(content="x")
        assert c.metadata == {}
