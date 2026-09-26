from __future__ import annotations
import asyncio, json, urllib.request, urllib.error
from .base import LLMProvider, LLMResponse

class OllamaError(RuntimeError): pass
class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, think: bool=False): self.base_url=base_url.rstrip("/"); self.model=model; self.think=think
    async def complete(self, messages: list[dict[str,str]], *, temperature: float=.3) -> LLMResponse:
        payload={"model":self.model,"messages":messages,"stream":False,"options":{"temperature":temperature}}
        # Ollama safely ignores unsupported options only on versions that support them.
        if self.think is False: payload["think"]=False
        def request():
            req=urllib.request.Request(self.base_url+"/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
            try:
                with urllib.request.urlopen(req,timeout=60) as response: return json.loads(response.read())
            except (urllib.error.URLError, TimeoutError) as exc: raise OllamaError(f"Ollama is unavailable at {self.base_url}: {exc.reason if hasattr(exc,'reason') else exc}") from exc
            except urllib.error.HTTPError as exc:
                body=exc.read().decode(errors="replace"); raise OllamaError(f"Ollama request failed ({exc.code}): {body}") from exc
        data=await asyncio.to_thread(request); return LLMResponse(data.get("message",{}).get("content","").strip(),data.get("model",self.model),data)
