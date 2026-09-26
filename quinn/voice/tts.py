from __future__ import annotations
from abc import ABC,abstractmethod
class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self,text:str)->bytes: ...
class TerminalTTS(TTSProvider):
    async def synthesize(self,text:str)->bytes: return b"" # UI renders text; no audio emitted.
