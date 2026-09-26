from __future__ import annotations
import logging, re
from datetime import datetime, timezone
from quinn.config import Settings
from .context import ContextManager, ConversationState
from .personality import load_system_prompt, concise
from .router import Router, Route
from .permissions import PermissionManager, PermissionLevel
from .memory import MemoryManager
from quinn.llm.ollama import OllamaProvider, OllamaError
from quinn.search.web import DuckDuckGoSearch
from quinn.tools.registry import Tool, ToolRegistry
from quinn.tools.location import MockLocationProvider
from quinn.tools.messaging import MockMessagingProvider
from quinn.tools.recording import RecordingService
from quinn.tools.emergency import EmergencyService

log=logging.getLogger("quinn.core")
class Quinn:
    def __init__(self, settings:Settings):
        self.settings=settings; settings.data_dir.mkdir(parents=True,exist_ok=True)
        self.context=ContextManager(settings.conversation_timeout_seconds); self.router=Router(); self.memory=MemoryManager(settings.data_dir,settings.memory_enabled)
        self.llm=OllamaProvider(settings.ollama_base_url,settings.ollama_model,settings.ollama_think); self.search=DuckDuckGoSearch()
        self.permissions=PermissionManager(); self.location=MockLocationProvider(); self.messaging=MockMessagingProvider(); self.recording=RecordingService(settings.data_dir/"recordings")
        self.emergency=EmergencyService(self.location,self.messaging,settings.emergency_contacts,settings.emergency_location_interval_seconds); self.tools=ToolRegistry(self.permissions); self._register_tools()
    def _register_tools(self):
        self.tools.register(Tool("web_search","Search public web for current information",{"query":"string"},self._web_tool,PermissionLevel.READ_ONLY))
        self.tools.register(Tool("get_location","Get current device location",{},self.location.get_current_location,PermissionLevel.READ_ONLY))
        self.tools.register(Tool("send_message","Send a message to a named recipient",{"recipient":"string","message":"string"},self.messaging.send_message,PermissionLevel.EXTERNAL_ACTION))
        self.tools.register(Tool("activate_emergency","Activate emergency mode",{},self.emergency.activate,PermissionLevel.CRITICAL)); self.tools.register(Tool("stop_emergency","Stop emergency mode",{},self.emergency.stop,PermissionLevel.CRITICAL))
        self.tools.register(Tool("start_recording","Start explicit environmental recording",{},self.recording.start,PermissionLevel.EXTERNAL_ACTION)); self.tools.register(Tool("stop_recording","Stop environmental recording",{},self.recording.stop,PermissionLevel.EXTERNAL_ACTION))
    async def _web_tool(self,query:str)->dict:
        try:return {"ok":True,"results":[r.as_dict() for r in await self.search.search(query)]}
        except Exception as exc:return {"ok":False,"error":f"Web lookup failed: {exc}"}
    def _system(self, memories:list[dict], web:list[dict]) -> str:
        dynamic=f"\nTime: {datetime.now(timezone.utc).isoformat()}\nState: {self.context.state.value}\nAvailable tools: {self.tools.available()}\nMemories: {[m['content'] for m in memories]}\nWeb results: {web}"
        return load_system_prompt()+dynamic
    async def _answer(self,text:str,web:list[dict]|None=None)->str:
        memories=self.memory.retrieve(text,self.settings.memory_top_k); messages=[{"role":"system","content":self._system(memories,web or [])},*self.context.history(),{"role":"user","content":text}]
        try:
            log.info("model_request",extra={"model":self.settings.ollama_model}); response=await self.llm.complete(messages); return concise(response.content)
        except OllamaError as exc:
            return "I can’t reach Ollama right now. Start the local server, then try again."
    def _maybe_remember(self, text: str) -> None:
        """Conservative implicit memory: stable preferences and active projects only."""
        if not self.settings.memory_enabled:
            return
        lowered = text.lower().strip()
        category = None
        if re.match(r"i (prefer|like|dislike|always want|never want)\b", lowered): category = "preference"
        elif re.match(r"i(?:'m| am) building\b", lowered): category = "project"
        if category:
            try: self.memory.add(text, category=category, importance=.55, source="conversation")
            except Exception: log.exception("implicit_memory_error")
    async def handle(self,text:str)->str:
        text=text.strip()
        if not text:return ""
        if self.context.timed_out(): self.context.transition(ConversationState.IDLE); log.info("conversation_timeout")
        route=self.router.classify(text); log.info("route",extra={"route":route.route.value})
        # deterministic state controls
        q=text.lower()
        if route.route is Route.SYSTEM_COMMAND:
            if any(x in q for x in ("wait","give me a minute","hold on","don't answer yet")):
                self.context.transition(ConversationState.WAITING); return "Okay. I’ll wait."
            if any(x in q for x in ("keep listening","continue")) or q=="quinn": self.context.transition(ConversationState.CONVERSATION); return "I’m listening."
            self.context.transition(ConversationState.IDLE); return "Going quiet."
        if self.context.state is ConversationState.WAITING and q not in {"quinn","continue","keep listening"}: return "I’m waiting. Say “continue” when you’re ready."
        if route.route is Route.EMERGENCY:
            if route.detail=="start":
                if not self.settings.emergency_enabled:return "Emergency mode is disabled in configuration."
                result=await self.tools.execute("activate_emergency",explicit=True); self.context.transition(ConversationState.EMERGENCY); return result["message"]
            if route.detail=="stop":
                result=await self.tools.execute("stop_emergency",explicit=True); self.context.transition(ConversationState.CONVERSATION); return result["message"]
            return "Do you want me to activate emergency mode? Say “I’m in an emergency” to confirm."
        if route.route is Route.MEMORY_WRITE:
            content=re.sub(r"^\s*(remember|don't forget|keep in mind)\s*(that)?\s*", "", text, flags=re.I).strip(" .")
            if not content:return "What should I remember?"
            item=self.memory.add(content,category="general"); return "Got it." if not item.get("duplicate") else "I already have that." 
        if route.route is Route.MEMORY_READ:
            memories=self.memory.retrieve(text,self.settings.memory_top_k)
            if not memories:return "I don’t have anything relevant saved yet."
            return " ".join(m["content"] for m in memories[:3])
        if route.route is Route.MEMORY_DELETE:
            target=re.sub(r"^\s*forget\s*(that)?\s*", "", text, flags=re.I).strip(" .")
            if not target:return "What should I forget?"
            return "Forgot it." if self.memory.delete(target) else "I couldn’t find a matching memory."
        if route.route is Route.TOOL:
            if "start recording" in q:return (await self.tools.execute("start_recording",explicit=True))["message"]
            if "stop recording" in q:return (await self.tools.execute("stop_recording",explicit=True))["message"]
            if "location" in q:
                r=await self.tools.execute("get_location"); return r["message"]
            match=re.match(r"\s*(?:text|message|tell)\s+(\w+)\s+(.+)",text,re.I)
            if match:
                r=await self.tools.execute("send_message",{"recipient":match.group(1),"message":match.group(2)},explicit=True); return r["message"]
        web=[]
        if route.route in {Route.SEARCH,Route.CURRENT_INFORMATION}:
            if not self.settings.web_search_enabled:return "Web search is disabled."
            result=await self._web_tool(text); 
            if not result["ok"]: return result["error"]
            web=result["results"]
            if not web:return "I couldn’t find current results for that."
        self.context.transition(ConversationState.CONVERSATION); self.context.add("user",text); answer=await self._answer(text,web); self.context.add("assistant",answer); self._maybe_remember(text); return answer
    async def shutdown(self):
        if self.emergency.active: await self.emergency.stop()
    def status(self)->dict: return {"state":self.context.state.value,"model":self.settings.ollama_model,"memory":self.settings.memory_enabled,"web_search":self.settings.web_search_enabled,"emergency":self.emergency.status()}
