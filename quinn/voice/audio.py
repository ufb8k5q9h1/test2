from __future__ import annotations
from abc import ABC,abstractmethod
class AudioTransport(ABC):
    @abstractmethod
    async def send_audio(self,audio:bytes)->None: ...
    @abstractmethod
    async def receive_audio(self)->bytes: ...
