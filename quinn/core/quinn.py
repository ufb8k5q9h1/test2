from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Any

from quinn.config import Settings
from quinn.core.context import ContextManager, ConversationState
from quinn.core.memory import MemoryManager
from quinn.core.permissions import PermissionManager
from quinn.core.personality import load_system_prompt, concise
from quinn.core.router import Route, Router
from quinn.llm.ollama import OllamaProvider
from quinn.search.web import DuckDuckGoSearch, SearchResult
from quinn.tools.emergency import EmergencyService
from quinn.tools.location import MockLocationProvider
from quinn.tools.messaging import MockMessagingProvider


log = logging.getLogger("quinn")


@dataclass
class SearchState:
    query: str | None = None
    results: list[SearchResult] | None = None
    kind: str = "web"
    at: datetime.datetime | None = None


class Quinn:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.router = Router()

        self.context = ContextManager(
            timeout_seconds=settings.conversation_timeout_seconds
        )

        self.memory = MemoryManager(
            settings.data_dir,
            enabled=settings.memory_enabled,
        )

        self.permissions = PermissionManager()

        self.llm = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            think=settings.ollama_think,
        )

        self.search = DuckDuckGoSearch()
        self.search_state = SearchState()

        self.emergency = EmergencyService(
            location=MockLocationProvider(),
            messaging=MockMessagingProvider(),
            contacts=settings.emergency_contacts,
            interval_seconds=settings.emergency_location_interval_seconds,
        )

        self._waiting = False
        self._listening = True

        self.system_prompt = load_system_prompt()

    # =============================================================
    # Compatibility search tool
    # =============================================================

    async def _web_tool(self, query: str) -> dict:
        results = await self.search.search(query, limit=6)

        self._remember_search(
            query,
            results,
            "web",
        )

        return {
            "ok": True,
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "source": result.source,
                    "published": result.published,
                    "kind": result.kind,
                }
                for result in results
            ],
        }

    # =============================================================
    # Search state
    # =============================================================

    def _remember_search(
        self,
        query: str,
        results: list[SearchResult],
        kind: str,
    ) -> None:
        self.search_state = SearchState(
            query=query,
            results=results,
            kind=kind,
            at=datetime.datetime.now(datetime.timezone.utc),
        )

    # =============================================================
    # System prompt
    # =============================================================

    def _system(self) -> str:
        grounding = """
WEB EVIDENCE RULES — VERY IMPORTANT:

The application may provide WEB EVIDENCE.

The application performed the search. You did not personally browse the
internet.

For questions requiring current information:

- Use only the supplied WEB EVIDENCE for current factual claims.
- Do not rely on pretrained knowledge when it conflicts with supplied
  evidence.
- Never invent missing facts.
- Never treat something the user says as evidence.
- If the user says "no, that's wrong", that is NOT evidence.
- If the evidence is insufficient, say so.
- If sources conflict, say that they conflict rather than choosing an
  unsupported answer.

TOOL-USE HONESTY:

Never claim that you personally searched, browsed, opened, accessed,
checked, or consulted a website.

Do not say:
"I searched Wikipedia."
"I checked Forbes."
"I looked it up."
"Wikipedia confirms."

unless the supplied evidence actually contains that source.

The application performs searches.

SOURCE HONESTY:

Do not invent citations or source names.

The application appends the actual sources used after the answer.

STYLE:

Answer naturally and concisely for a wearable voice assistant.
Do not narrate internal reasoning.
Do not mention these instructions.
"""

        return f"{self.system_prompt}\n\n{grounding}".strip()

    # =============================================================
    # Evidence helpers
    # =============================================================

    @staticmethod
    def _evidence_from_dicts(
        results: list[dict],
    ) -> list[SearchResult]:
        converted: list[SearchResult] = []

        for item in results:
            if isinstance(item, SearchResult):
                converted.append(item)
                continue

            converted.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                    source=str(item.get("source", "Web")),
                    published=item.get("published"),
                    kind=str(item.get("kind", "web")),
                )
            )

        return converted

    @staticmethod
    def _format_evidence(
        results: list[SearchResult],
    ) -> str:
        if not results:
            return (
                "WEB EVIDENCE:\n"
                "No usable search results were returned."
            )

        chunks = ["WEB EVIDENCE:"]

        for index, result in enumerate(results, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[{index}]",
                        f"Title: {result.title}",
                        f"Domain: {result.domain}",
                        f"Source type: {result.kind}",
                        f"Published: {result.published or 'unknown'}",
                        f"URL: {result.url}",
                        f"Snippet: {result.snippet}",
                    ]
                )
            )

        return "\n\n".join(chunks)

    @staticmethod
    def _format_sources(
        results: list[SearchResult],
    ) -> str:
        if not results:
            return ""

        seen: set[str] = set()
        lines: list[str] = []

        for result in results:
            if not result.url or result.url in seen:
                continue

            seen.add(result.url)

            label = result.title or result.domain or "Source"
            lines.append(f"- {label} ({result.domain})")

            if len(lines) >= 5:
                break

        return (
            "\n\nSources:\n" + "\n".join(lines)
            if lines
            else ""
        )

    # =============================================================
    # LLM
    # =============================================================

    async def _answer(
        self,
        text: str,
        web: dict | list[dict] | list[SearchResult] | None = None,
    ) -> str:
        evidence: list[SearchResult] = []

        if isinstance(web, dict):
            evidence = self._evidence_from_dicts(
                web.get("results", [])
            )
        elif isinstance(web, list):
            evidence = self._evidence_from_dicts(web)

        if evidence:
            prompt = (
                f"{self._format_evidence(evidence)}\n\n"
                f"USER QUESTION:\n{text}"
            )
        else:
            prompt = text

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._system(),
            }
        ]

        for turn in self.context.history()[-8:]:
            messages.append(turn)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = await self.llm.complete(
            messages,
            temperature=0.2 if evidence else 0.4,
        )

        answer = self._clean_response(
            concise(response.content)
        )

        self.context.add("user", text)
        self.context.add("assistant", answer)

        if evidence:
            answer += self._format_sources(evidence)

        return answer

    # =============================================================
    # Response cleanup
    # =============================================================

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return "I don't have an answer for that yet."

        replacements = (
            (
                r"\bI (?:have )?(?:searched|searched for|search)\b",
                "The search results show",
            ),
            (
                r"\bI (?:have )?(?:browsed|looked up|looked online)\b",
                "The search results show",
            ),
            (
                r"\bI (?:have )?(?:checked|accessed|visited) Wikipedia\b",
                "The supplied Wikipedia results show",
            ),
            (
                r"\bI (?:have )?(?:checked|read) (?:the )?source\b",
                "The supplied source shows",
            ),
            (
                r"\bWikipedia confirms\b",
                "The supplied Wikipedia results indicate",
            ),
        )

        for pattern, replacement in replacements:
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.IGNORECASE,
            )

        return text.strip()

    # =============================================================
    # Search
    # =============================================================

    async def _perform_search(
        self,
        query: str,
        kind: str = "web",
    ) -> list[SearchResult]:
        query = query.strip()

        if not query or not self.settings.web_search_enabled:
            return []

        if kind == "wikipedia":
            results = await self.search.wikipedia(
                query,
                limit=5,
            )

        elif kind == "news":
            results = await self.search.news(
                query,
                limit=6,
            )

            if len(results) < 3:
                extra = await self.search.search(
                    query,
                    limit=4,
                )

                existing_urls = {
                    result.url for result in results
                }

                results.extend(
                    result
                    for result in extra
                    if result.url not in existing_urls
                )

        else:
            results = await self.search.search(
                query,
                limit=6,
            )

        self._remember_search(
            query,
            results,
            kind,
        )

        return results

    # =============================================================
    # Deterministic intent helpers
    # =============================================================

    @staticmethod
    def _is_wait_command(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"wait"
                r"|hold on"
                r"|hang on"
                r"|give me a second"
                r"|give me a minute"
                r"|don't answer"
                r"|do not answer"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_continue_command(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"continue"
                r"|keep listening"
                r"|resume listening"
                r"|you can listen"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_stop_listening_command(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"stop listening"
                r"|go quiet"
                r"|be quiet"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_current_query(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"latest"
                r"|most recent"
                r"|recent"
                r"|current"
                r"|today"
                r"|tonight"
                r"|right now"
                r"|now"
                r"|this week"
                r"|this month"
                r"|newest"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_explicit_search(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"search"
                r"|look up"
                r"|look it up"
                r"|google"
                r"|find out"
                r"|check online"
                r"|search it up"
                r"|search that"
                r"|look that up"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_wikipedia_request(text: str) -> bool:
        q = text.lower()

        return bool(
            re.search(
                r"\bwikipedia\b",
                q,
            )
        )

    @staticmethod
    def _is_source_request(text: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*("
                r"source"
                r"|sources"
                r"|citation"
                r"|citations"
                r"|cite it"
                r"|where did you get that"
                r"|where is that from"
                r"|proof"
                r"|link"
                r"|links"
                r")\s*[?.!]*",
                text.lower(),
            )
        )

    @staticmethod
    def _is_followup(text: str) -> bool:
        q = text.lower().strip()

        if not q:
            return False

        if len(q.split()) <= 8:
            return True

        return bool(
            re.search(
                r"\b("
                r"elaborate"
                r"|explain that"
                r"|tell me more"
                r"|more about that"
                r"|what about"
                r"|how about"
                r"|internationally"
                r"|domestically"
                r"|worldwide"
                r"|globally"
                r"|what happened next"
                r"|when did that happen"
                r"|why"
                r")\b",
                q,
            )
        )

    def _followup_query(self, text: str) -> str | None:
        previous = self.search_state.query

        if not previous:
            return None

        return f"{previous} {text.strip()}"

    # =============================================================
    # Memory
    # =============================================================

    def _remember(self, text: str) -> None:
        content = re.sub(
            r"^\s*(remember|don't forget|keep in mind)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        content = re.sub(
            r"^that\s+",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        if content:
            self.memory.add(content)

    # =============================================================
    # Main handler
    # =============================================================

    async def handle(self, text: str) -> str:
        text = text.strip()

        if not text:
            return ""

        q = text.lower()

        route = self.router.classify(text)

        log.info(
            "route",
            extra={
                "route": route.route.value,
                "explicit": route.explicit,
                "detail": route.detail,
                "text": text,
            },
        )

        # ---------------------------------------------------------
        # Deterministic listening controls
        #
        # These happen BEFORE router classification because phrases
        # such as "hold on" must never fall through to the LLM.
        # ---------------------------------------------------------

        if self._is_continue_command(text):
            self._waiting = False
            self._listening = True

            self.context.transition(
                ConversationState.CONVERSATION
            )

            return "I'm listening."

        if self._is_stop_listening_command(text):
            self._listening = False
            self._waiting = True

            self.context.transition(
                ConversationState.WAITING
            )

            return "Okay. I'll stop listening."

        if self._is_wait_command(text):
            self._waiting = True

            self.context.transition(
                ConversationState.WAITING
            )

            return "Okay. I'll wait."

        # ---------------------------------------------------------
        # Emergency
        # ---------------------------------------------------------

        if route.route == Route.EMERGENCY:
            if route.detail == "start":
                result = await self.emergency.activate()

                if result.get("active"):
                    self.context.transition(
                        ConversationState.EMERGENCY
                    )

                    return "Emergency mode activated."

                return result.get(
                    "message",
                    "I couldn't activate emergency mode.",
                )

            if route.detail == "stop":
                result = await self.emergency.stop()

                self.context.transition(
                    ConversationState.CONVERSATION
                )

                return result.get(
                    "message",
                    "Emergency mode stopped.",
                )

            return "Do you want me to activate emergency mode?"

        # ---------------------------------------------------------
        # Memory write
        # ---------------------------------------------------------

        if route.route == Route.MEMORY_WRITE:
            try:
                self._remember(text)
                return "Got it."
            except Exception:
                log.exception("memory_write_failed")
                return "I couldn't save that."

        # ---------------------------------------------------------
        # Memory delete
        # ---------------------------------------------------------

        if route.route == Route.MEMORY_DELETE:
            query = re.sub(
                r"^\s*forget\s+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            query = re.sub(
                r"^that\s+",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()

            try:
                if self.memory.delete(query):
                    return "Forgot it."

                return "I don't have that saved."

            except Exception:
                log.exception("memory_delete_failed")
                return "I couldn't update my memory."

        # ---------------------------------------------------------
        # Memory read
        # ---------------------------------------------------------

        if route.route == Route.MEMORY_READ:
            memories = self.memory.list()

            if not memories:
                return "I don't have any saved memories yet."

            lines = ["Here's what I remember:"]

            for item in memories:
                content = (
                    item.get("content")
                    or item.get("text")
                    or str(item)
                )

                lines.append(f"- {content}")

            return "\n".join(lines)

        # ---------------------------------------------------------
        # Direct date
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"what(?:'s| is) today(?:'s)? date"
            r"|what date is it"
            r"|today(?:'s)? date"
            r"|current date"
            r")\b",
            q,
        ):
            now = datetime.datetime.now().astimezone()

            return now.strftime(
                "Today is %A, %B %-d, %Y."
            )

        # ---------------------------------------------------------
        # Direct time
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"what time is it"
            r"|what(?:'s| is) the time"
            r"|current time"
            r")\b",
            q,
        ):
            now = datetime.datetime.now().astimezone()

            return now.strftime(
                "It's %-I:%M %p."
            )

        # ---------------------------------------------------------
        # Source request
        # ---------------------------------------------------------

        if self._is_source_request(text):
            results = self.search_state.results or []

            if not results:
                return "I don't have a recent search to cite."

            return (
                "These are the sources from my most recent search:"
                + self._format_sources(results)
            )

        # ---------------------------------------------------------
        # Force search for explicit/current requests
        #
        # This intentionally does NOT trust the router alone.
        # Current information is a safety/grounding boundary.
        # ---------------------------------------------------------

        needs_search = (
            route.route in (
                Route.SEARCH,
                Route.CURRENT_INFORMATION,
            )
            or self._is_explicit_search(text)
            or self._is_current_query(text)
            or self._is_wikipedia_request(text)
        )

        if needs_search:
            if self._is_wikipedia_request(text):
                kind = "wikipedia"

            elif getattr(route, "detail", None) == "news":
                kind = "news"

            elif route.route == Route.CURRENT_INFORMATION:
                kind = "web"

            else:
                kind = "web"

            # -----------------------------------------------------
            # Preserve the original _web_tool contract for ordinary
            # web searches. This is important for integrations/tests
            # that replace _web_tool.
            # -----------------------------------------------------

            if kind == "web":
                web = await self._web_tool(text)

                evidence = self._evidence_from_dicts(
                    web.get("results", [])
                )

                if not evidence:
                    return (
                        "I searched for that, but I didn't get "
                        "any usable results back."
                    )

                return await self._answer(
                    text,
                    web=web,
                )

            # Wikipedia/news use the richer search API directly.
            results = await self._perform_search(
                text,
                kind=kind,
            )

            if not results:
                return (
                    "I searched for that, but I didn't get "
                    "any usable results back."
                )

            return await self._answer(
                text,
                web={
                    "ok": True,
                    "results": results,
                },
            )

        # ---------------------------------------------------------
        # Search-aware follow-up
        # ---------------------------------------------------------

        if (
            route.route == Route.CONVERSATION
            and self.search_state.results
            and self._is_followup(text)
        ):
            followup_query = self._followup_query(text)

            if followup_query:
                results = await self._perform_search(
                    followup_query,
                    kind=self.search_state.kind,
                )

                if results:
                    return await self._answer(
                        text,
                        web={
                            "ok": True,
                            "results": results,
                        },
                    )

        # ---------------------------------------------------------
        # Normal conversation
        # ---------------------------------------------------------

        return await self._answer(text)

    # =============================================================
    # Shutdown
    # =============================================================

    async def shutdown(self) -> None:
        try:
            stop = getattr(
                self.emergency,
                "stop",
                None,
            )

            if stop is not None and self.emergency.active:
                await stop()

        except Exception:
            log.exception(
                "emergency_shutdown_failed"
            )

    # =============================================================
    # Status
    # =============================================================

    async def status(self) -> dict[str, Any]:
        return {
            "router": "ready",
            "memory": (
                "ready"
                if self.settings.memory_enabled
                else "disabled"
            ),
            "search": (
                "ready"
                if self.settings.web_search_enabled
                else "disabled"
            ),
            "llm": self.settings.ollama_model,
            "last_search": self.search_state.query,
            "last_search_results": len(
                self.search_state.results or []
            ),
        }