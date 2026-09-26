from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
class RecordingService:
    def __init__(self, directory:Path): self.directory=directory; self.active=False; self.started_at=None
    async def start(self)->dict:
        if self.active:return {"ok":True,"already_active":True,"mock":True,"message":"Mock environmental recording is already active."}
        self.directory.mkdir(parents=True,exist_ok=True); self.active=True; self.started_at=datetime.now(timezone.utc).isoformat()
        return {"ok":True,"mock":True,"recording":False,"message":"Mock recording armed; no microphone audio is captured on this laptop prototype."}
    async def stop(self)->dict:
        was=self.active; self.active=False
        return {"ok":True,"mock":True,"recording":False,"message":"Mock recording stopped." if was else "No recording was active."}
