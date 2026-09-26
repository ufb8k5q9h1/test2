from __future__ import annotations
from dataclasses import dataclass
from typing import Awaitable, Callable, Any
from quinn.core.permissions import PermissionLevel, PermissionManager
@dataclass
class Tool:
    name: str; description: str; parameters: dict; handler: Callable[..., Awaitable[dict]]; permission: PermissionLevel
class ToolRegistry:
    def __init__(self, permissions: PermissionManager): self.permissions=permissions; self._tools:dict[str,Tool]={}
    def register(self, tool: Tool): self._tools[tool.name]=tool
    def available(self): return [{"name":t.name,"description":t.description,"parameters":t.parameters,"permission":t.permission.name} for t in self._tools.values()]
    async def execute(self,name:str,arguments:dict|None=None,*,explicit=False)->dict:
        tool=self._tools.get(name)
        if not tool: return {"ok":False,"error":"Unknown tool"}
        if not self.permissions.allowed(tool.permission,explicit): return {"ok":False,"error":"Permission denied: explicit user intent required."}
        try: return await tool.handler(**(arguments or {}))
        except Exception as exc: return {"ok":False,"error":str(exc)}
