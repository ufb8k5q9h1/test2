import pytest

from quinn.config import Settings
from quinn.core.context import ContextManager, ConversationState
from quinn.core.personality import concise
from quinn.core.router import Router, Route
from quinn.core.memory import MemoryManager
from quinn.core.permissions import PermissionManager, PermissionLevel
from quinn.core.quinn import Quinn
from quinn.search.web import SearchResult
from quinn.tools.location import MockLocationProvider
from quinn.tools.messaging import MockMessagingProvider


def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        web_search_enabled=True,
        emergency_location_interval_seconds=3600,
    )


def test_concise_response_configuration():
    assert (
        concise(
            "One. Two. Three. Four."
        )
        == "One. Two. Three."
    )


def test_router_classification_and_no_search_for_simple_commands():
    r = Router()

    assert (
        r.classify(
            "remember that I prefer tea"
        ).route
        is Route.MEMORY_WRITE
    )

    assert (
        r.classify(
            "what do you remember about tea?"
        ).route
        is Route.MEMORY_READ
    )

    assert (
        r.classify(
            "what do you even know about me?"
        ).route
        is Route.MEMORY_READ
    )

    assert (
        r.classify(
            "how much do you know about me?"
        ).route
        is Route.MEMORY_READ
    )

    assert (
        r.classify(
            "tell me what you remember about me"
        ).route
        is Route.MEMORY_READ
    )

    assert (
        r.classify(
            "forget that I prefer tea"
        ).route
        is Route.MEMORY_DELETE
    )

    assert (
        r.classify(
            "what is 2+2?"
        ).route
        is Route.CONVERSATION
    )

    assert (
        r.classify(
            "what's the latest Python version?"
        ).route
        is Route.CURRENT_INFORMATION
    )

    assert (
        r.classify(
            "what was Kendrick's last feature?"
        ).route
        is Route.CURRENT_INFORMATION
    )

    assert (
        r.classify(
            "search this for me"
        ).route
        is Route.SEARCH
    )


def test_context_timeout_and_waiting_state():
    c = ContextManager(
        timeout_seconds=10
    )

    c.transition(
        ConversationState.CONVERSATION
    )

    assert c.timed_out(
        c.last_activity + 11
    )

    c.transition(
        ConversationState.WAITING
    )

    assert not c.timed_out(
        c.last_activity + 1000
    )


def test_memory_add_retrieve_delete(tmp_path):
    m = MemoryManager(tmp_path)

    item = m.add(
        "The user is building Quinn for AR glasses",
        category="project",
    )

    assert (
        m.retrieve(
            "AR glasses project"
        )[0]["id"]
        == item["id"]
    )

    assert m.delete(
        item["id"]
    )

    assert not m.retrieve(
        "AR glasses"
    )


@pytest.mark.asyncio
async def test_mock_location_and_messaging():
    location = (
        await MockLocationProvider()
        .get_current_location()
    )

    assert location["mock"]
    assert not location["available"]
    assert location["latitude"] is None

    message = (
        await MockMessagingProvider()
        .send_message(
            "Alex",
            "Hi",
        )
    )

    assert message["mock"]
    assert not message["sent"]


def test_permission_enforcement():
    p = PermissionManager()

    assert p.allowed(
        PermissionLevel.READ_ONLY
    )

    assert not p.allowed(
        PermissionLevel.CRITICAL
    )

    assert p.allowed(
        PermissionLevel.CRITICAL,
        explicit=True,
    )


@pytest.mark.asyncio
async def test_wait_stop_emergency_and_ambiguous_language(
    tmp_path,
):
    q = Quinn(
        settings(tmp_path)
    )

    assert (
        "wait"
        in (
            await q.handle(
                "hold on"
            )
        ).lower()
    )

    assert (
        q.context.state
        is ConversationState.WAITING
    )

    assert (
        "listening"
        in (
            await q.handle(
                "continue"
            )
        ).lower()
    )

    assert (
        "activate"
        in (
            await q.handle(
                "this game is an emergency lol"
            )
        ).lower()
    )

    assert not q.emergency.active

    out = await q.handle(
        "I'm in an emergency"
    )

    assert q.emergency.active
    assert "activated" in out.lower()

    out = await q.handle(
        "stop emergency mode"
    )

    assert not q.emergency.active
    assert "stopped" in out.lower()


@pytest.mark.asyncio
async def test_memory_and_web_routing_without_real_network(
    tmp_path,
    monkeypatch,
):
    q = Quinn(
        settings(tmp_path)
    )

    assert (
        await q.handle(
            "remember that I prefer concise replies"
        )
        == "Got it."
    )

    assert (
        "concise"
        in (
            await q.handle(
                "what do you remember about preferences?"
            )
        ).lower()
    )

    assert (
        await q.handle(
            "forget that I prefer concise replies"
        )
        == "Forgot it."
    )

    calls = []

    async def fake_web(query):
        calls.append(query)

        return {
            "ok": True,
            "results": [
                {
                    "title": "Python",
                    "url": "https://python.org",
                    "snippet": "Current version test result",
                    "source": "Web",
                    "kind": "web",
                }
            ],
        }

    q._web_tool = fake_web

    async def fake_answer(
        text,
        web=None,
    ):
        return "Current result."

    q._answer = fake_answer

    assert (
        await q.handle(
            "What is the latest Python version?"
        )
        == "Current result."
    )

    assert calls

    calls.clear()

    await q.handle(
        "what is 2+2?"
    )

    assert not calls


@pytest.mark.asyncio
async def test_memory_question_never_searches(
    tmp_path,
):
    q = Quinn(
        settings(tmp_path)
    )

    calls = []

    async def fake_web(query):
        calls.append(query)
        return {
            "ok": True,
            "results": [],
        }

    q._web_tool = fake_web

    q.memory.add(
        "I prefer concise replies",
        category="preference",
    )

    async def fake_answer(
        text,
        web=None,
    ):
        raise AssertionError(
            "Memory query reached the LLM"
        )

    q._answer = fake_answer

    result = await q.handle(
        "What do you even know about me?"
    )

    assert "concise" in result.lower()
    assert not calls


@pytest.mark.asyncio
async def test_search_followup_resolves_subject(
    tmp_path,
):
    q = Quinn(
        settings(tmp_path)
    )

    calls = []

    async def fake_web(query):
        calls.append(query)

        return {
            "ok": True,
            "results": [
                {
                    "title": "Kendrick Lamar",
                    "url": "https://example.com/kendrick",
                    "snippet": (
                        "Kendrick Lamar latest feature "
                        "and September 11 2026 release."
                    ),
                    "source": "Web",
                    "kind": "web",
                }
            ],
        }

    q._web_tool = fake_web

    async def fake_answer(
        text,
        web=None,
    ):
        return "Verified result."

    q._answer = fake_answer

    await q.handle(
        "What was Kendrick's last feature?"
    )

    calls.clear()

    await q.handle(
        "what was it?"
    )

    assert calls
    assert "Kendrick Lamar" in calls[-1]


@pytest.mark.asyncio
async def test_correction_rechecks_previous_subject(
    tmp_path,
):
    q = Quinn(
        settings(tmp_path)
    )

    calls = []

    async def fake_web(query):
        calls.append(query)

        return {
            "ok": True,
            "results": [
                {
                    "title": "Kendrick Lamar",
                    "url": "https://example.com/kendrick",
                    "snippet": (
                        "Kendrick Lamar latest feature."
                    ),
                    "source": "Web",
                    "kind": "web",
                }
            ],
        }

    q._web_tool = fake_web

    async def fake_answer(
        text,
        web=None,
    ):
        return "Checked."

    q._answer = fake_answer

    await q.handle(
        "What was Kendrick's last feature?"
    )

    calls.clear()

    await q.handle(
        "That's wrong on so many levels."
    )

    assert calls

    # The correction must NOT become the search query.
    assert "wrong on so many levels" not in (
        calls[-1].lower()
    )

    assert "kendrick" in (
        calls[-1].lower()
    )


@pytest.mark.asyncio
async def test_user_date_hint_creates_clean_followup_query(
    tmp_path,
):
    q = Quinn(
        settings(tmp_path)
    )

    calls = []

    async def fake_web(query):
        calls.append(query)

        return {
            "ok": True,
            "results": [
                {
                    "title": "Kendrick Lamar",
                    "url": "https://example.com/kendrick",
                    "snippet": (
                        "Kendrick Lamar feature "
                        "September 11 2026."
                    ),
                    "source": "Web",
                    "kind": "web",
                }
            ],
        }

    q._web_tool = fake_web

    async def fake_answer(
        text,
        web=None,
    ):
        return "Checked."

    q._answer = fake_answer

    await q.handle(
        "What was Kendrick's last feature?"
    )

    calls.clear()

    await q.handle(
        "Kendrick's last feature was on "
        "September 11 2026. What was it?"
    )

    assert calls

    query = calls[-1].lower()

    assert "kendrick" in query
    assert "september" in query
    assert "2026" in query


def test_followup_detection_does_not_treat_arithmetic_as_search():
    assert not Quinn._is_followup(
        "what is 2+2?"
    )

    assert Quinn._is_followup(
        "what was it?"
    )

    assert Quinn._is_followup(
        "that's wrong"
    )

    assert Quinn._is_followup(
        "tell me more about that"
    )