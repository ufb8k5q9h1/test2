from __future__ import annotations
from .embeddings import cosine
class Retriever:
    def __init__(self, store, embedder): self.store,self.embedder=store,embedder
    def search(self, query: str, top_k: int=5) -> list[dict]:
        needle=self.embedder.embed(query); ranked=[]
        for item in self.store.all():
            score=cosine(needle,item["embedding"])
            if score > 0: ranked.append(({k:v for k,v in item.items() if k != "embedding"},score))
        return [{**item,"relevance":round(score,3)} for item,score in sorted(ranked,key=lambda x:x[1],reverse=True)[:top_k]]
