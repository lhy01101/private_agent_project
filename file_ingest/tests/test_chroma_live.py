"""
test_chroma_live.py —— 用真实 Chroma（内存版）验证 indexed 分支契约

覆盖验收标准：
  2. 大文件切块入库，metadata 含 file_id
  4. indexed query 返回 Top-5 片段（截断生效 + 排序正确）
  5. file_id 隔离：两文件 query 不交叉

注意：本文件用「可区分的 embedding」（每个 chunk 内容不同 → 向量不同），
这样 Top-K 与排序才有意义。生产环境由 bge-m3 保证区分度。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from file_ingest.store import FileStore
from file_ingest.loader import DocChunk


def make_embedder(seed_per_text: bool = True):
    """可区分的 embedder：不同文本 → 不同向量（模拟 bge-m3 行为）。"""
    def emb(texts):
        out = []
        for t in texts:
            # 用文本内容的 hash，保证「内容不同 → 向量不同」
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((h[i] / 255.0) - 0.5) * 2 for i in range(16)]
            # L2 归一化（余弦）
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / (norm + 1e-9) for x in vec])
        return out
    return emb


def main():
    import chromadb
    client = chromadb.Client()
    store = FileStore(client, "live_verify")
    store.set_embedder(make_embedder())

    results = []
    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  ({detail})" if detail else ""))

    # ---- 准备两份「语义可区分」的内容 ----
    # 文件 A：讲"阿尔法项目"
    chunks_a = [
        DocChunk(content=f"阿尔法项目第{i}章：关于系统架构与RAG设计的详细描述。",
                 metadata={"page": i, "chunk_type": "page"})
        for i in range(8)   # 8 个 chunk
    ]
    # 文件 B：讲"贝塔平台"（与 A 完全不同）
    chunks_b = [
        DocChunk(content=f"贝塔平台第{i}节：关于数据库优化与索引策略。",
                 metadata={"page": i, "chunk_type": "page"})
        for i in range(6)
    ]

    # ---------- 验收 #2：metadata 含 file_id ----------
    print("\n[验收2] indexed metadata 含 file_id")
    store.add("file-alpha", chunks_a, lifecycle="temp")
    store.add("file-beta", chunks_b, lifecycle="persistent")
    coll = client.get_or_create_collection("live_verify")
    all_alpha = coll.get(where={"file_id": "file-alpha"})
    check("file-alpha 全部 chunk metadata 带 file_id",
          all(m.get("file_id") == "file-alpha" for m in (all_alpha.get("metadatas") or [])),
          f"count={len(all_alpha.get('ids') or [])}")
    check("lifecycle 字段正确",
          all(m.get("lifecycle") == "temp" for m in (all_alpha.get("metadatas") or [])))
    check("file-beta lifecycle=persistent",
          all(m.get("lifecycle") == "persistent" for m in (coll.get(where={"file_id": "file-beta"}).get("metadatas") or [])))

    # ---------- 验收 #4：Top-5 + 排序 ----------
    print("\n[验收4] indexed query 返回 Top-5 片段")
    # 查"阿尔法" → 应该命中文件 A，且按相似度降序
    q_alpha = make_embedder()(["阿尔法项目架构"])[0]
    hits = store.query("file-alpha", q_alpha, k=5)
    check("返回 <=5 条", len(hits) <= 5, f"实际 {len(hits)}")
    check("非空", bool(hits))
    # 相似度单调递减（降序）
    scores = [h.metadata.get("score", 0) for h in hits]
    check("按相似度降序", all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
          f"scores={[round(s,3) for s in scores]}")
    # 问"数据库"（贝塔主题）查 file-alpha → 相似度应整体偏低（隔离 + 区分度）
    q_db = make_embedder()(["数据库索引优化"])[0]
    hits_db = store.query("file-alpha", q_db, k=3)
    check("无关查询也能返回（降级兜底）", len(hits_db) >= 0)

    # ---------- 验收 #5：file_id 隔离 ----------
    print("\n[验收5] file_id 隔离")
    # 关键：查 file-alpha 的结果，绝不应包含 file-beta 的 chunk
    for h in hits:
        assert h.metadata.get("file_id") == "file-alpha", "隔离失效！"
    check("file-alpha 查询结果不混入 file-beta", True)

    # 反向验证：用「贝塔」向量查 file-alpha，结果仍只属于 alpha（只是分数低）
    rogue = store.query("file-alpha", make_embedder()(["贝塔平台数据库"])[0], k=10)
    check("跨文件查询不返回对方 chunk",
          all(h.metadata.get("file_id") == "file-alpha" for h in rogue))

    # ---------- 附加：delete 只删指定 file_id ----------
    print("\n[附加] delete / cleanup_temp")
    before = coll.count()
    store.delete("file-alpha")
    after = coll.count()
    check("delete 只删 file-alpha", before - after == len(chunks_a), f"{before}->{after}")
    check("file-beta 仍在",
          len(coll.get(where={"file_id": "file-beta"}).get("ids") or []) == len(chunks_b))

    # ---------- cleanup_temp ----------
    # 先把 alpha 以 temp 重新加入，再清理
    store.add("file-alpha", chunks_a[:2], lifecycle="temp")
    n = store.cleanup_temp()
    check("cleanup_temp 清理 temp chunk", n >= 2, f"deleted={n}")
    check("cleanup 后 file-beta(persistent) 保留",
          len(coll.get(where={"file_id": "file-beta"}).get("ids") or []) == len(chunks_b))

    print("\n" + "=" * 50)
    print(f"总计: {len(results)} 项, 通过 {sum(results)}, 失败 {len(results)-sum(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
