"""
conftest.py —— 为验收测试准备测试文件 + 轻量 Chroma（内存版）

由于完整依赖（chromadb / pdfplumber / tree-sitter 等）在沙盒中可能不全，
这里做「依赖可选 + 降级」测试：能跑的真实跑，缺依赖的验证接口契约。
"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tmp_files(tmp_path_factory) -> dict[str, Path]:
    """生成各格式测试文件。缺依赖时用纯文本 .txt 占位验证流程。"""
    base = tmp_path_factory.mktemp("files")
    files: dict[str, Path] = {}

    # .py 代码文件（正则/tree-sitter 切块用）
    py = base / "sample.py"
    py.write_text(
        '"""模块说明：示例"""\n'
        "\n"
        "def hello(name):\n"
        '    """打招呼"""\n'
        '    return f"hi {name}"\n'
        "\n"
        "\n"
        "class Calculator:\n"
        '    """简单计算器"""\n'
        "\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def subtract(self, a, b):\n"
        "        return a - b\n"
        "\n"
        "\n"
        "def main():\n"
        "    c = Calculator()\n"
        "    print(c.add(1, 2))\n"
    )
    files["py"] = py

    # .txt 作为通用文本（验证 TextLoader / 大文件阈值分支）
    txt = base / "note.txt"
    txt.write_text("这是第一行。\n这是第二行，用于测试全文缓存。\n" * 50)
    files["txt"] = txt

    # .docx / .pptx / .pdf：仅在对应依赖可用时生成真实文件
    files["docx"] = base / "report.docx"
    files["pptx"] = base / "deck.pptx"
    files["pdf"] = base / "paper.pdf"

    return files


@pytest.fixture(scope="session")
def chroma_client():
    """优先真实 chromadb，不可用则用内存 dict 自造一个兼容壳（仅供无依赖环境跑契约测试）。"""
    try:
        import chromadb  # noqa
        client = chromadb.Client()
        yield client
        return
    except Exception:
        pass

    # ---- 降级：内存版 Chroma 壳，满足 store.py 的 add/query/delete/get 接口 ----
    class _MemCollection:
        def __init__(self, name):
            self.name = name
            self._docs: dict[str, dict] = {}

        def get_or_create_collection(self, name, metadata=None):
            return self

        def count(self):
            return len(self._docs)

        def add(self, ids, embeddings, documents, metadatas):
            for i, id_ in enumerate(ids):
                self._docs[id_] = {
                    "embedding": embeddings[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                }

        def query(self, query_embeddings, where, n_results=5):
            # 按 where 过滤 + 余弦相似度排序（降级的极简实现）
            q = query_embeddings[0]
            cand = [
                (id_, d) for id_, d in self._docs.items()
                if all(d["metadata"].get(k) == v for k, v in where.items())
            ]
            if not cand:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            def sim(item):
                e = item[1]["embedding"]
                dot = sum(a * b for a, b in zip(q, e))
                na = sum(a * a for a in q) ** 0.5
                nb = sum(a * a for a in e) ** 0.5
                return dot / (na * nb + 1e-9)

            cand.sort(key=sim, reverse=True)
            top = cand[:n_results]
            return {
                "documents": [[c[1]["document"] for c in top]],
                "metadatas": [[c[1]["metadata"] for c in top]],
                "distances": [[1.0 - sim(c) for c in top]],
            }

        def delete(self, where=None, ids=None):
            if ids:
                for id_ in ids:
                    self._docs.pop(id_, None)
            elif where:
                to_del = [
                    id_ for id_, d in self._docs.items()
                    if all(d["metadata"].get(k) == v for k, v in where.items())
                ]
                for id_ in to_del:
                    self._docs.pop(id_, None)

        def get(self, where=None, limit=None):
            if not where:
                items = list(self._docs.values())
            else:
                items = [
                    d for d in self._docs.values()
                    if all(d["metadata"].get(k) == v for k, v in where.items())
                ]
            if limit:
                items = items[:limit]
            return {
                "ids": list(self._docs.keys())[:len(items)],
                "documents": [d["document"] for d in items],
                "metadatas": [d["metadata"] for d in items],
            }

    class _MemClient:
        def __init__(self):
            self._collections: dict[str, _MemCollection] = {}

        def get_or_create_collection(self, name, metadata=None):
            if name not in self._collections:
                self._collections[name] = _MemCollection(name)
            return self._collections[name]

    yield _MemClient()


@pytest.fixture(scope="session")
def embed_fn():
    """bge-m3 占位：沙盒无模型时用随机固定维度向量，足以验证「隔离/排序」契约。"""
    import hashlib

    DIM = 8

    def _fn(texts):
        out = []
        for t in texts:
            # 用文本 hash 生成确定性向量，保证相同文本向量相同（可验证排序）
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((h[i] / 255.0) - 0.5) * 2 for i in range(DIM)]
            # L2 归一化（store.query 用余弦）
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / (norm + 1e-9) for x in vec])
        return out

    return _fn


@pytest.fixture
def configured(chroma_client, embed_fn):
    """装配 FileStore + embedder，返回 (store, reset)。"""
    from file_ingest import FileStore, configure

    store = FileStore(chroma_client, collection_name="test_files")
    configure(store, embed_fn)
    yield store
    # 清理：每个测试后清集合
    try:
        chroma_client.get_or_create_collection("test_files").delete()
    except Exception:
        pass
