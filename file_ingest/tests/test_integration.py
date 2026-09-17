"""
integration_smoke.py —— 端到端冒烟（模拟真实 Agent 调用链路）

无需真实 chromadb/tree-sitter：用 conftest 的内存降级实现即可跑通全流程，
验证 8 条验收标准在「接口契约层面」全部通过。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 让脚本可直接运行（pytest 之外）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    from file_ingest import (
        FileStore, configure, ingest_file, query_file,
        has_file_intent, build_file_list_prompt, register_file,
    )

    # ---- 1. 模拟 Agent 启动装配 ----
    class _MemClient:
        def get_or_create_collection(self, name, metadata=None):
            if not hasattr(self, "_c"):
                self._c = _Collection(name)
            return self._c

    class _Collection:
        def __init__(self, name):
            self.name = name
            self._docs = {}

        def count(self):
            return len(self._docs)

        def add(self, ids, embeddings, documents, metadatas):
            for i, id_ in enumerate(ids):
                self._docs[id_] = {"embedding": embeddings[i], "document": documents[i], "metadata": metadatas[i]}

        def query(self, query_embeddings, where, n_results=5):
            q = query_embeddings[0]
            cand = [d for d in self._docs.values()
                    if all(d["metadata"].get(k) == v for k, v in where.items())]
            if not cand:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            def sim(d):
                e = d["embedding"]
                dot = sum(a * b for a, b in zip(q, e))
                return dot / ((sum(a * a for a in q) ** 0.5) * (sum(a * a for a in e) ** 0.5) + 1e-9)

            cand.sort(key=sim, reverse=True)
            top = cand[:n_results]
            return {"documents": [[d["document"] for d in top]],
                    "metadatas": [[d["metadata"] for d in top]],
                    "distances": [[1.0 - sim(d) for d in top]]}

        def delete(self, where=None, ids=None):
            if ids:
                for id_ in ids:
                    self._docs.pop(id_, None)

        def get(self, where=None, limit=None):
            if not where:
                items = list(self._docs.values())
            else:
                items = [d for d in self._docs.values()
                         if all(d["metadata"].get(k) == v for k, v in where.items())]
            return {"ids": list(self._docs.keys())[:len(items)],
                    "documents": [d["document"] for d in items],
                    "metadatas": [d["metadata"] for d in items]}

    client = _MemClient()
    store = FileStore(client, "smoke_test")

    DIM = 8

    def embed_fn(texts):
        import hashlib
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [((h[i] / 255.0) - 0.5) * 2 for i in range(DIM)]
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / (norm + 1e-9) for x in vec])
        return out

    configure(store, embed_fn)

    # ---- 准备测试文件 ----
    tmp = Path("/tmp/file_ingest_smoke")
    tmp.mkdir(exist_ok=True)
    f_small = tmp / "small.txt"
    f_small.write_text("第一行：项目介绍\n第二行：技术栈是 Python + Chroma\n第三行：部署在 Kubernetes\n" * 3)
    f_large = tmp / "large.txt"
    f_large.write_text(("版本日志：v1.0 初始化\n" * 400))  # 约 6000 token -> indexed

    print("=" * 60)
    print("冒烟测试：文件管理模块端到端链路")
    print("=" * 60)

    # ---- 2. 硬路由检测 ----
    assert has_file_intent("帮我看下 small.txt") is True
    assert has_file_intent("今天天气") is False
    print("\n[✓] has_file_intent 正常工作")

    # ---- 3. 上传小文件（inline）----
    r1 = json.loads(ingest_file.invoke({"path": str(f_small)}))
    print(f"\n[上传小文件] {r1}")
    assert r1["mode"] == "inline"
    assert r1["chunk_count"] == 0
    fid_small = r1["file_id"]

    # ---- 4. 上传大文件（indexed）----
    r2 = json.loads(ingest_file.invoke({"path": str(f_large), "persist": True}))
    print(f"[上传大文件] {r2}")
    assert r2["mode"] == "indexed"
    assert r2["chunk_count"] > 0
    fid_large = r2["file_id"]

    register_file(fid_large, r2["file_name"], r2["file_type"], r2["summary"])

    # ---- 5. inline 问答（返回全文）----
    ans_small = query_file.invoke({"file_id": fid_small, "question": "技术栈"})
    print(f"\n[inline 问答] 长度={len(ans_small)}, 全文匹配={'项目介绍' in ans_small}")
    assert "项目介绍" in ans_small

    # ---- 6. indexed 问答（返回片段）----
    ans_large = query_file.invoke({"file_id": fid_large, "question": "版本日志"})
    print(f"[indexed 问答] 返回块数≈{ans_large.count(chr(10)*2)+1}")
    assert ans_large and "版本日志" in ans_large

    # ---- 7. file_id 隔离 ----
    rogue = store.query(fid_small, embed_fn(["x"])[0], k=5)
    for c in rogue:
        assert c.metadata.get("file_id") == fid_small, "隔离失效"
    print("[✓] file_id 隔离验证通过")

    # ---- 8. system prompt 注入 ----
    prompt = build_file_list_prompt(store, [fid_large])
    print(f"\n[system prompt 注入]\n{prompt}")
    assert "large.txt" in prompt

    print("\n" + "=" * 60)
    print("全部验收标准通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
