from __future__ import annotations
from pathlib import Path
from quinn.rag.store import LocalVectorStore
from quinn.rag.embeddings import HashEmbedding, cosine
from quinn.rag.retrieval import Retriever

class MemoryManager:
    categories={"preference","person","project","fact","instruction","event","general"}
    def __init__(self, data_dir: Path, enabled=True):
        self.enabled=enabled; self.embedder=HashEmbedding(); self.store=LocalVectorStore(data_dir / "memory" / "quinn.sqlite3"); self.retriever=Retriever(self.store,self.embedder)
    def add(self, content: str, category="general", importance=.7, source="user") -> dict:
        if not self.enabled: raise RuntimeError("Memory is disabled")
        content=content.strip(); existing=self.retriever.search(content,1)
        if existing and existing[0]["relevance"] > .96: return {**existing[0],"duplicate":True}
        return self.store.add(content,self.embedder.embed(content),category if category in self.categories else "general",importance,source)
    def retrieve(self, query: str, top_k=5) -> list[dict]: return self.retriever.search(query,top_k) if self.enabled else []
    def delete(self, query_or_id: str) -> bool:
        if self.store.delete(query_or_id): return True
        found=self.retrieve(query_or_id,1); return bool(found and found[0]["relevance"]>.4 and self.store.delete(found[0]["id"]))
    def list(self) -> list[dict]: return [{k:v for k,v in x.items() if k != "embedding"} for x in self.store.all()]
