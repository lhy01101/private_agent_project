"""
smoke.py —— 脱离 pytest 的独立冒烟测试

只测「纯 Python 可实现」的部分：loader 正则切块、router、tools 的 inline 分支。
chromadb/tree-sitter 相关在真实环境跑 test_acceptance.py。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from file_ingest.loader import get_loader, CodeRegexLoader, TextLoader
from file_ingest.router import has_file_intent


def main():
    results: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  ({detail})" if detail else ""))

    print("\n[1] has_file_intent")
    check("含扩展名 → True", has_file_intent("帮我看下 report.pdf") is True)
    check("attachments → True", has_file_intent({"attachments": ["/tmp/a.docx"]}) is True)
    check("file_id → True", has_file_intent({"file_id": "abc-123"}) is True)
    check("纯闲聊 → False", has_file_intent("今天天气怎么样") is False)
    check("写诗 → False", has_file_intent("帮我写首诗") is False)

    print("\n[2] CodeRegexLoader (.py 切块)")
    with tempfile.TemporaryDirectory() as d:
        py = Path(d) / "sample.py"
        py.write_text(
            '"""模块说明"""\n\n\ndef hello(name):\n    return f"hi {name}"\n\n\n'
            "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
        )
        chunks = CodeRegexLoader(".py").load(str(py))
        names = [c.metadata.get("name") for c in chunks]
        check("拆出 hello", "hello" in names)
        check("拆出 Calculator", "Calculator" in names)
        check("带 lineno", all("lineno" in c.metadata for c in chunks))
        # 按源码顺序：hello(1) 在 Calculator(7) 之前 → lineno 单调递增
        by_name = {c.metadata.get("name"): c.metadata["lineno"] for c in chunks}
        check("lineno 单调递增", by_name.get("hello", 0) < by_name.get("Calculator", 0))

    print("\n[3] TextLoader（未知扩展名兜底）")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "weird.xyz"
        f.write_text("hello world")
        chunks = get_loader(str(f)).load(str(f))
        check("不报错且非空", bool(chunks) and chunks[0].content == "hello world")

    print("\n[4] get_loader 分发")
    with tempfile.TemporaryDirectory() as d:
        for ext in [".py", ".go", ".ts", ".java", ".js"]:
            f = Path(d) / f"a{ext}"
            f.write_text(f"func main() {{}}")
            check(f"{ext} 有 loader", get_loader(str(f)) is not None)

    print("\n[5] tools 的 inline 分支（mock Chroma/embedder）")
    # 用一个自造的 store + embedder，验证小文件走 inline、全文进缓存
    from file_ingest.tools import configure, ingest_file, query_file, _INLINE_CACHE, _STORE, _EMBED_FN

    class _Coll:
        def __init__(self):
            self.docs = {}
        def count(self):
            return len(self.docs)
        def add(self, ids, embeddings, documents, metadatas):
            for i, id_ in enumerate(ids):
                self.docs[id_] = {"document": documents[i], "metadata": metadatas[i]}
        def query(self, qe, where, n_results=5):
            fid = where.get("file_id")
            hit = [d for d in self.docs.values() if d["metadata"].get("file_id") == fid]
            if not hit:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            return {"documents": [[h["document"] for h in hit[:n_results]]],
                    "metadatas": [[h["metadata"] for h in hit[:n_results]]],
                    "distances": [[0.5] * min(n_results, len(hit))]}
        def delete(self, where=None, ids=None):
            pass
        def get(self, where=None, limit=None):
            return {"ids": [], "documents": [], "metadatas": []}

    class _Client:
        def get_or_create_collection(self, name, metadata=None):
            if not hasattr(self, "c"):
                self.c = _Coll()
            return self.c

    def embed_fn(texts):
        import hashlib
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([((h[i] / 255.0) - 0.5) * 2 for i in range(8)])
        return out

    store = type("S", (), {"client": _Client(), "collection_name": "t",
                           "collection": None, "_embedder": embed_fn})()
    # 直接构造 FileStore 会触发 chromadb 导入（在 property 里惰性），我们 mock 掉 _collection
    from file_ingest.store import FileStore
    real_store = FileStore(_Client(), "t")
    real_store._collection = _Coll()
    configure(real_store, embed_fn)

    with tempfile.TemporaryDirectory() as d:
        txt = Path(d) / "note.txt"
        txt.write_text("第一行：项目介绍\n第二行：技术栈 Python\n第三行：部署 K8s\n" * 3)
        res = json.loads(ingest_file.invoke({"path": str(txt)}))
        check("返回 mode=inline", res["mode"] == "inline", str(res))
        check("chunk_count=0", res["chunk_count"] == 0)
        check("file_id 进缓存", res["file_id"] in _INLINE_CACHE)
        # query inline → 返回全文
        full = _INLINE_CACHE[res["file_id"]]
        ans = query_file.invoke({"file_id": res["file_id"], "question": "技术栈"})
        check("inline query 返回全文", ans.strip() == full.strip() and "项目介绍" in ans)
        # query 不存在的 fid
        err = json.loads(query_file.invoke({"file_id": "nope", "question": "x"}))
        check("未知 fid 返回 error", "error" in err)

    print("\n" + "=" * 50)
    failed = [r for r in results if not r[1]]
    for name, ok, detail in failed:
        print(f"  ✗ {name}  {detail}")
    print(f"\n总计: {len(results)} 项, 通过 {len(results)-len(failed)}, 失败 {len(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
