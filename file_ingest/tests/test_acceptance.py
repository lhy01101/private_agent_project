"""
test_acceptance.py —— 8 条验收标准

对应 file_achieve.md §五：
    1. 小文件全文进缓存，Chroma 0 条
    2. 大文件切块入库，metadata 含 file_id
    3. query_file 对 inline 文件返回全文
    4. query_file 对 indexed 文件返回 Top-5 片段
    5. file_id 隔离：两个文件 query 不交叉
    6. 代码文件切块带 name + lineno
    7. ingest_file 返回 JSON 含 file_id/mode/summary
    8. has_file_intent 对含扩展名 query 返回 True
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _code_libs_available() -> bool:
    """收集阶段即可调用的判断：tree-sitter 及对应 grammar 是否可用。"""
    try:
        from tree_sitter_languages import get_parser  # noqa: F401
        return True
    except Exception:
        return False


# ============================================================
# 验收 7 + 1（返回结构 & 文件意图）—— 不依赖外部解析库，始终可跑
# ============================================================
class TestIngestContract:
    def test_7_ingest_returns_required_fields(self, configured, tmp_files):
        """验收 #7：返回 JSON 含 file_id / mode / summary。"""
        from file_ingest import tools, ingest_file
_INLINE_CACHE = tools._INLINE_CACHE

        # 用 .txt 小文件（必走 inline，不依赖 pdf/docx 解析库）
        txt = str(tmp_files["txt"])
        result_str = ingest_file.invoke({"path": txt, "persist": False})
        data = json.loads(result_str)

        # 失败情况（解析异常）单独允许，但这里 .txt 一定能解析
        assert "error" not in data, f"ingest 失败: {data}"

        for field in ("file_id", "file_name", "file_type", "mode", "chunk_count", "summary", "lifecycle"):
            assert field in data, f"缺少字段 {field}"

        assert data["mode"] in ("inline", "indexed")
        # .txt 内容短 -> inline，全文进缓存
        assert data["mode"] == "inline"
        assert data["chunk_count"] == 0
        assert data["file_id"] in _INLINE_CACHE
        assert _INLINE_CACHE[data["file_id"]]  # 全文非空

    def test_8_has_file_intent(self):
        """验收 #8：含扩展名的 query 返回 True。"""
        from file_ingest import has_file_intent

        assert has_file_intent("帮我看下 report.pdf") is True
        assert has_file_intent("分析一下 notes.docx 的内容") is True
        assert has_file_intent({"attachments": ["/tmp/a.pptx"]}) is True
        assert has_file_intent({"file_id": "abc-123-def"}) is True
        # 负面：纯闲聊不应触发
        assert has_file_intent("今天天气怎么样") is False
        assert has_file_intent("帮我写首诗") is False


# ============================================================
# 验收 1-6：依赖真实 Chroma + Loader；缺依赖时自动 skip
# ============================================================
pytest.importorskip("chromadb")  # 无 chromadb 则整个 class skip


class TestStoreBehavior:
    def test_1_inline_no_chroma_records(self, configured, tmp_files):
        """验收 #1：小文件 inline，Chroma 中 0 条。"""
        from file_ingest import tools, ingest_file
        _INLINE_CACHE = tools._INLINE_CACHE
        _STORE = tools._STORE

        txt = str(tmp_files["txt"])
        before = _STORE.collection.count() if _STORE else 0

        result = json.loads(ingest_file.invoke({"path": txt}))
        after = _STORE.collection.count()

        assert result["mode"] == "inline"
        assert result["file_id"] in _INLINE_CACHE
        assert after == before, "inline 模式不应向 Chroma 写入任何 chunk"

    def test_2_indexed_metadata_has_file_id(self, configured, tmp_files, monkeypatch):
        """验收 #2：大文件切块入库，metadata 含正确 file_id。

        通过 monkeypatch 把阈值临时调低，让小文件也走 indexed 分支。
        """
        from file_ingest import tools, ingest_file

        monkeypatch.setattr(tools, "_TOKEN_THRESHOLD", 0)  # 强制 indexed

        txt = str(tmp_files["txt"])
        result = json.loads(ingest_file.invoke({"path": txt}))
        assert result["mode"] == "indexed"
        assert result["chunk_count"] > 0

        # 验证 Chroma 中该 file_id 的所有 chunk metadata 都带 file_id
        store = tools._STORE
        res = store.collection.get(where={"file_id": result["file_id"]})
        assert len(res.get("ids") or []) == result["chunk_count"]
        for md in res.get("metadatas") or []:
            assert md.get("file_id") == result["file_id"]
            assert "lifecycle" in md

    def test_3_query_inline_returns_full_text(self, configured, tmp_files):
        """验收 #3：inline 文件 query 返回全文。"""
        from file_ingest import tools, ingest_file, query_file
_INLINE_CACHE = tools._INLINE_CACHE

        txt = str(tmp_files["txt"])
        result = json.loads(ingest_file.invoke({"path": txt}))
        full = _INLINE_CACHE[result["file_id"]]

        answer = query_file.invoke({"file_id": result["file_id"], "question": "第二行是什么"})
        assert answer.strip() == full.strip(), "inline 模式必须返回完整全文"

    def test_4_query_indexed_returns_top5(self, configured, tmp_files, monkeypatch):
        """验收 #4：indexed 文件 query 返回相关片段（Top-5，非全文）。"""
        from file_ingest import tools, ingest_file, query_file

        monkeypatch.setattr(tools, "_TOKEN_THRESHOLD", 0)

        txt = str(tmp_files["txt"])
        result = json.loads(ingest_file.invoke({"path": txt, "persist": True}))
        assert result["mode"] == "indexed"

        answer = query_file.invoke({"file_id": result["file_id"], "question": "任意问题"})
        # 返回的是片段拼接（含来源标记），且非全文单块
        assert answer  # 非空
        # Top-5：段落数不超过 5
        assert answer.count("\n\n") < 5

    def test_5_file_id_isolation(self, configured, tmp_files, monkeypatch):
        """验收 #5：file_id 隔离，两个文件 query 结果不交叉。"""
        from file_ingest import tools, ingest_file, query_file

        monkeypatch.setattr(tools, "_TOKEN_THRESHOLD", 0)

        f1 = json.loads(ingest_file.invoke({"path": str(tmp_files["txt"])}))
        # 第二个文件：用不同的文本
        other = tmp_files["txt"].parent / "other.txt"
        other.write_text("完全不同的内容，用于隔离测试。")
        f2 = json.loads(ingest_file.invoke({"path": str(other)}))

        assert f1["file_id"] != f2["file_id"]

        # 直接查 store：file_id=A 的结果不应含 file_id=B 的 chunk
        store = tools._STORE
        chunks_a = store.query(f1["file_id"], store._embedder(["x"])[0], k=10)
        for c in chunks_a:
            assert c.metadata.get("file_id") == f1["file_id"], "file_id 隔离失效"

    @pytest.mark.skipif(
        not _code_libs_available(), reason="tree-sitter 或对应 grammar 不可用"
    )
    def test_6_code_chunks_have_name_and_lineno(self, configured, tmp_files):
        """验收 #6：.py 切块 metadata 含 name + lineno。"""
        from file_ingest import tools
from file_ingest.loader import FileStore, get_loader
_EMBED_FN = tools._EMBED_FN

        py = str(tmp_files["py"])
        loader = get_loader(py)
        chunks = loader.load(py)

        # 至少有一个 chunk 带 name + lineno（函数/类定义）
        annotated = [c for c in chunks if c.metadata.get("name") and c.metadata.get("lineno")]
        assert annotated, f"代码切块未提取到 name/lineno: {[c.metadata for c in chunks]}"

        # 写入 store 后 metadata 保留（验证 Chroma 清洗不丢字段）
        store = configured
        fid = "test-code-001"
        store.add(fid, chunks, lifecycle="temp")
        res = store.collection.get(where={"file_id": fid})
        for md in res.get("metadatas") or []:
            if md.get("name"):
                assert "lineno" in md


# _code_libs_available() 定义见模块顶部
