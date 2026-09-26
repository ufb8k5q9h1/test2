from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from time import monotonic

class ConversationState(str, Enum):
    IDLE = "IDLE"; AWAKE = "AWAKE"; CONVERSATION = "CONVERSATION"; WAITING = "WAITING"; EMERGENCY = "EMERGENCY"

@dataclass
class Turn:
    role: str; content: str; at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ContextManager:
    timeout_seconds: int = 180
    max_turns: int = 12
    state: ConversationState = ConversationState.IDLE
    turns: list[Turn] = field(default_factory=list)
    last_activity: float = field(default_factory=monotonic)
    recent_audio_metadata: dict | None = None

    def transition(self, state: ConversationState) -> None:
        self.state = state; self.last_activity = monotonic()

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role, content)); self.turns = self.turns[-self.max_turns:]; self.last_activity = monotonic()

    def timed_out(self, now: float | None = None) -> bool:
        return self.state in {ConversationState.AWAKE, ConversationState.CONVERSATION} and (now or monotonic()) - self.last_activity >= self.timeout_seconds

    def history(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None: self.turns.clear()
