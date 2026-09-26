from __future__ import annotations
import json, sqlite3
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

class LocalVectorStore:
    """SQLite local vector database; embeddings live locally and never leave the device."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, embedding TEXT NOT NULL, created_at TEXT, updated_at TEXT, category TEXT, importance REAL, source TEXT)"); self.conn.commit()
    def add(self, content: str, embedding: list[float], category="general", importance=0.5, source="user") -> dict:
        now=datetime.now(timezone.utc).isoformat(); item={"id":str(uuid4()),"content":content,"created_at":now,"updated_at":now,"category":category,"importance":importance,"source":source}
        self.conn.execute("INSERT INTO memories VALUES (:id,:content,:embedding,:created_at,:updated_at,:category,:importance,:source)", {**item,"embedding":json.dumps(embedding)}); self.conn.commit(); return item
    def all(self) -> list[dict]:
        rows=self.conn.execute("SELECT id,content,embedding,created_at,updated_at,category,importance,source FROM memories").fetchall(); keys=("id","content","embedding","created_at","updated_at","category","importance","source")
        return [{**dict(zip(keys,row)),"embedding":json.loads(row[2])} for row in rows]
    def delete(self, memory_id: str) -> bool:
        cur=self.conn.execute("DELETE FROM memories WHERE id=?",(memory_id,)); self.conn.commit(); return bool(cur.rowcount)
    def update(self, memory_id: str, content: str, embedding: list[float]) -> bool:
        cur=self.conn.execute("UPDATE memories SET content=?,embedding=?,updated_at=? WHERE id=?",(content,json.dumps(embedding),datetime.now(timezone.utc).isoformat(),memory_id)); self.conn.commit(); return bool(cur.rowcount)
