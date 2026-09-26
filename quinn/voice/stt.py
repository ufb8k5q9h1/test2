from __future__ import annotations
from abc import ABC,abstractmethod
class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self,audio:bytes)->str: ...
class TerminalSTT(STTProvider):
    async def transcribe(self,audio:bytes)->str: raise NotImplementedError("Terminal mode receives text directly.")
