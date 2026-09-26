from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    model: str
    raw: dict | None = None

class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str,str]], *, temperature: float=.3) -> LLMResponse: ...
