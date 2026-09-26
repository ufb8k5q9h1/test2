from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
class LocationProvider(ABC):
    @abstractmethod
    async def get_current_location(self)->dict: ...
class MockLocationProvider(LocationProvider):
    async def get_current_location(self)->dict:
        return {"available":False,"mock":True,"latitude":None,"longitude":None,"accuracy_m":None,"timestamp":datetime.now(timezone.utc).isoformat(),"message":"No real location provider is configured."}
