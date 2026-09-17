"""
store.py —— FileStore：ChromaDB 封装

契约（file_achieve.md §3.2）：
- 复用现有 Chroma 实例（同一 chroma_client），不新建
- add：每个 chunk metadata 必须含 file_id
- query：必须用 where={"file_id": file_id} 过滤（file_id 隔离）
- lifecycle：temp / persistent
- embed_fn：与现有知识库一致的 bge-m3
"""
from __future__ import annotations

from typing import Callable, Optional
from .loader import DocChunk


# 默认 embedder 占位；运行时通过 set_embedder() 注入现有 bge-m3 函数
def _default_embedder(texts):
    raise RuntimeError("FileStore.embedder 未初始化：请调用 store.set_embedder(embed_fn)")


class FileStore:
    def __init__(self, chroma_client, collection_name: str, embed_fn: Optional[Callable] = None):
        """
        Args:
            chroma_client: 现有 ChromaDB client（复用，不新建）
            collection_name: 集合名（可与现有知识库共用或独立，由调用方决定）
            embed_fn: 接受 list[str] -> list[list[float]] 的 embedding 函数（bge-m3）
        """
        self.client = chroma_client
        self.collection_name = collection_name
        self._embedder: Callable = embed_fn or _default_embedder
        self._collection = None  # 延迟初始化

    # ---------- embedder 注入（支持运行时替换，便于测试 mock） ----------
    def set_embedder(self, embed_fn: Callable) -> None:
        """注入现有 bge-m3 embedding 函数。"""
        self._embedder = embed_fn

    @property
    def collection(self):
        """懒加载：避免导入 chromadb 时立即建连。"""
        if self._collection is None:
            import chromadb
            # get_or_create：复用现有 collection，符合"不新建"要求
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ---------- 写入 ----------
    def add(self, file_id: str, chunks: list[DocChunk], lifecycle: str) -> int:
        """
        将 chunks 写入 Chroma。每个 chunk metadata 强制注入 file_id / lifecycle / file_type。

        Returns:
            实际写入条数
        """
        if not chunks:
            return 0

        contents = [c.content for c in chunks]
        embeddings = self._embedder(contents)

        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        for c in chunks:
            md = dict(c.metadata)
            md["file_id"] = file_id
            md["lifecycle"] = lifecycle
            # Chroma metadata 只支持 str/int/float/bool，做一次清洗
            metadatas.append({k: _chroma_value(v) for k, v in md.items()})

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        return len(chunks)

    # ---------- 查询 ----------
    def query(self, file_id: str, query_embedding: list[float], k: int = 5) -> list[DocChunk]:
        """
        对该 file_id 做向量检索，where 过滤保证 file_id 隔离（契约核心）。

        Returns:
            Top-k DocChunk 列表（按相似度降序，含来源 metadata）
        """
        if self.collection.count() == 0:
            return []

        res = self.collection.query(
            query_embeddings=[query_embedding],
            where={"file_id": file_id},   # ★ 隔离：只查该文件的 chunk
            n_results=k,
        )
        chunks: list[DocChunk] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        for doc, md, dist in zip(docs, metas, distances):
            md = dict(md or {})
            md["score"] = 1.0 - float(dist)  # 余弦距离 -> 相似度
            chunks.append(DocChunk(content=doc, metadata=md))
        return chunks

    # ---------- 删除 / 清理 ----------
    def delete(self, file_id: str) -> None:
        """删除该 file_id 的全部 chunk。"""
        self.collection.delete(where={"file_id": file_id})

    def cleanup_temp(self) -> int:
        """
        清理所有 lifecycle='temp' 的 chunk。返回删除的 file_id 数量（近似）。

        注意：Chroma 按 where 删除是按 chunk 粒度，此处返回被影响的 chunk 数。
        """
        try:
            # 先查出 temp chunk 的 id
            col = self.collection
            # 若集合为空直接返回
            if col.count() == 0:
                return 0
            # 用 get 取 temp 的所有 id
            res = col.get(where={"lifecycle": "temp"})
            ids = res.get("ids") or []
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception as e:
            # 部分旧版 chromadb 对 where 支持有限，降级为遍历 file_id 的备选在 delete() 之上
            raise RuntimeError(f"cleanup_temp 失败: {e}") from e


def _chroma_value(v):
    """Chroma metadata 仅支持基础标量类型。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
