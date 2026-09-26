from __future__ import annotations
import asyncio, logging, sys
from quinn.config import load_settings
from quinn.core.quinn import Quinn
def configure_logging(level): logging.basicConfig(level=getattr(logging,level.upper(),logging.INFO),format="%(asctime)s %(levelname)s %(name)s %(message)s")
async def run():
    settings=load_settings(); configure_logging(settings.log_level); quinn=Quinn(settings)
    print(f"Quinn online.\nModel: {settings.ollama_model}\nMemory: {'enabled' if settings.memory_enabled else 'disabled'}\nWeb search: {'enabled' if settings.web_search_enabled else 'disabled'}\nType '/exit' to quit.\n")
    try:
        while True:
            try:
                # Some Python/PTY combinations cannot read a pipe from an executor
                # thread. Keep interactive input non-blocking for the event loop,
                # while making redirected input reliable for automation.
                if sys.stdin.isatty(): text=await asyncio.to_thread(input,"You: ")
                else:
                    print("You: ",end="",flush=True); text=sys.stdin.readline().rstrip("\n")
            except EOFError:break
            if not text and not sys.stdin.isatty(): break
            if text.strip().lower() in {"/exit","exit","quit"}:break
            if text.strip()=="/status":print("Quinn:",quinn.status());continue
            if text.strip()=="/memory":
                items=quinn.memory.list(); print("Quinn:", "No saved memories." if not items else "\n".join(f"- {x['content']} ({x['id']})" for x in items));continue
            if text.strip()=="/clear":quinn.context.clear();print("Quinn: Conversation cleared.");continue
            response=await quinn.handle(text)
            if response:print("Quinn:",response)
    finally: await quinn.shutdown()
def main(): asyncio.run(run())
if __name__=="__main__":main()
