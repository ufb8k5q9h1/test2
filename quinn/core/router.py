from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re

class Route(str, Enum):
    CONVERSATION="conversation"; CURRENT_INFORMATION="current_information"; SEARCH="search"; TOOL="tool"; MEMORY_WRITE="memory_write"; MEMORY_READ="memory_read"; MEMORY_DELETE="memory_delete"; SYSTEM_COMMAND="system_command"; EMERGENCY="emergency"
@dataclass(frozen=True)
class RouteResult:
    route: Route; explicit: bool = False; detail: str = ""

class Router:
    current_terms = ("latest", "current", "today", "news", "weather", "price", "score", "schedule", "version", "recent")
    def classify(self, text: str) -> RouteResult:
        q = text.lower().strip()
        if re.search(r"\b(stop listening|go quiet|cancel|wait|give me a minute|hold on|don't answer yet|keep listening|continue)\b", q): return RouteResult(Route.SYSTEM_COMMAND, True)
        if re.search(r"\b(stop emergency|end emergency|cancel emergency)\b", q): return RouteResult(Route.EMERGENCY, True, "stop")
        if re.search(r"\b(i(?:'m| am) in an emergency|activate emergency|emergency mode)\b", q): return RouteResult(Route.EMERGENCY, True, "start")
        if "emergency" in q: return RouteResult(Route.EMERGENCY, False, "ambiguous")
        if re.match(r"\s*(remember|don't forget|keep in mind)\b", q): return RouteResult(Route.MEMORY_WRITE, True)
        if re.match(r"\s*forget\b", q): return RouteResult(Route.MEMORY_DELETE, True)
        if re.search(r"\b(what do you remember|remember about|my memories)\b", q): return RouteResult(Route.MEMORY_READ, True)
        if re.match(r"\s*(search|look up|google)\b", q): return RouteResult(Route.SEARCH, True)
        if re.match(r"\s*(text|message|tell)\s+\w+", q): return RouteResult(Route.TOOL, True, "message")
        if re.search(r"\b(start recording|stop recording|where am i|get location)\b", q): return RouteResult(Route.TOOL, True)
        if any(term in q for term in self.current_terms): return RouteResult(Route.CURRENT_INFORMATION)
        return RouteResult(Route.CONVERSATION)
