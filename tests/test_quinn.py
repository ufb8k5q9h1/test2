import asyncio
from pathlib import Path
import pytest
from quinn.config import Settings
from quinn.core.context import ContextManager, ConversationState
from quinn.core.personality import concise
from quinn.core.router import Router, Route
from quinn.core.memory import MemoryManager
from quinn.core.permissions import PermissionManager, PermissionLevel
from quinn.core.quinn import Quinn
from quinn.tools.location import MockLocationProvider
from quinn.tools.messaging import MockMessagingProvider

def settings(tmp_path): return Settings(data_dir=tmp_path, web_search_enabled=True, emergency_location_interval_seconds=3600)
def test_concise_response_configuration(): assert concise("One. Two. Three. Four.") == "One. Two. Three."
def test_router_classification_and_no_search_for_simple_commands():
    r=Router(); assert r.classify("remember that I prefer tea").route is Route.MEMORY_WRITE
    assert r.classify("what do you remember about tea?").route is Route.MEMORY_READ
    assert r.classify("forget that I prefer tea").route is Route.MEMORY_DELETE
    assert r.classify("what is 2+2?").route is Route.CONVERSATION
    assert r.classify("what's the latest Python version?").route is Route.CURRENT_INFORMATION
    assert r.classify("search this for me").route is Route.SEARCH
def test_context_timeout_and_waiting_state():
    c=ContextManager(timeout_seconds=10); c.transition(ConversationState.CONVERSATION); assert c.timed_out(c.last_activity+11)
    c.transition(ConversationState.WAITING); assert not c.timed_out(c.last_activity+1000)
def test_memory_add_retrieve_delete(tmp_path):
    m=MemoryManager(tmp_path); item=m.add("The user is building Quinn for AR glasses",category="project")
    assert m.retrieve("AR glasses project")[0]["id"] == item["id"]
    assert m.delete(item["id"]); assert not m.retrieve("AR glasses")
@pytest.mark.asyncio
async def test_mock_location_and_messaging():
    location=await MockLocationProvider().get_current_location(); assert location["mock"] and not location["available"] and location["latitude"] is None
    message=await MockMessagingProvider().send_message("Alex","Hi"); assert message["mock"] and not message["sent"]
def test_permission_enforcement():
    p=PermissionManager(); assert p.allowed(PermissionLevel.READ_ONLY); assert not p.allowed(PermissionLevel.CRITICAL); assert p.allowed(PermissionLevel.CRITICAL,explicit=True)
@pytest.mark.asyncio
async def test_wait_stop_emergency_and_ambiguous_language(tmp_path):
    q=Quinn(settings(tmp_path)); assert "wait" in (await q.handle("hold on")).lower(); assert q.context.state is ConversationState.WAITING
    assert "listening" in (await q.handle("continue")).lower(); assert "activate" in (await q.handle("this game is an emergency lol")).lower(); assert not q.emergency.active
    out=await q.handle("I'm in an emergency"); assert q.emergency.active and "activated" in out.lower()
    out=await q.handle("stop emergency mode"); assert not q.emergency.active and "stopped" in out.lower()
@pytest.mark.asyncio
async def test_memory_and_web_routing_without_real_network(tmp_path, monkeypatch):
    q=Quinn(settings(tmp_path)); assert await q.handle("remember that I prefer concise replies") == "Got it."
    assert "concise" in (await q.handle("what do you remember about preferences?")).lower()
    assert await q.handle("forget that I prefer concise replies") == "Forgot it."
    calls=[]
    async def fake_web(query): calls.append(query); return {"ok":True,"results":[{"title":"Python","url":"https://python.org","snippet":"Current version test result"}]}
    q._web_tool=fake_web
    async def fake_answer(text,web=None): return "Current result."
    q._answer=fake_answer
    assert await q.handle("What is the latest Python version?") == "Current result."; assert calls
    calls.clear(); await q.handle("what is 2+2?"); assert not calls
