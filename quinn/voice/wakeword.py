from __future__ import annotations
from abc import ABC,abstractmethod
class WakeWordProvider(ABC):
    @abstractmethod
    async def detected(self,audio:bytes)->bool: ...
class TextWakeWord(WakeWordProvider):
    async def detected(self,audio:bytes)->bool: return "hey quinn" in audio.decode(errors="ignore").lower()
