from __future__ import annotations
from enum import IntEnum

class PermissionLevel(IntEnum):
    READ_ONLY = 1; LOW_RISK = 2; EXTERNAL_ACTION = 3; CRITICAL = 4

class PermissionManager:
    """Terminal prototype approves safe operations; consequential operations need explicit intent."""
    def allowed(self, level: PermissionLevel, explicit: bool = False) -> bool:
        return level <= PermissionLevel.LOW_RISK or explicit
