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
    subject: str | None = None
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
    # Search tool
    # =============================================================

    async def _web_tool(self, query: str) -> dict:
        results = await self.search.search(
            query,
            limit=8,
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
        subject: str | None = None,
    ) -> None:
        if subject is None:
            subject = self._extract_search_subject(
                query,
                results,
            )

        self.search_state = SearchState(
            query=query,
            results=results,
            kind=kind,
            subject=subject,
            at=datetime.datetime.now(
                datetime.timezone.utc
            ),
        )

    # =============================================================
    # Search subject extraction
    # =============================================================

    @staticmethod
    def _extract_search_subject(
        query: str,
        results: list[SearchResult] | None = None,
    ) -> str | None:
        clean = query.strip()

        # Known entity aliases.
        if re.search(
            r"\bKendrick(?:'s)?\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return "Kendrick Lamar"

        # Search result evidence can reveal the canonical name.
        if results:
            combined = " ".join(
                f"{result.title} {result.snippet}"
                for result in results[:6]
            )

            if re.search(
                r"\bKendrick Lamar\b",
                combined,
                flags=re.IGNORECASE,
            ):
                return "Kendrick Lamar"

        # Generic proper-name extraction.
        clean = re.sub(
            r"^\s*(search|look up|google|find out|check online)\s+",
            "",
            clean,
            flags=re.IGNORECASE,
        )

        words = clean.split()
        proper: list[str] = []

        for word in words:
            stripped = re.sub(
                r"[^A-Za-z'-]",
                "",
                word,
            )

            if (
                stripped
                and stripped[0].isupper()
            ):
                proper.append(stripped)

                if len(proper) >= 4:
                    break

        if proper:
            return " ".join(proper)

        return None

    # =============================================================
    # Query cleanup
    # =============================================================

    @staticmethod
    def _clean_search_text(text: str) -> str:
        query = text.strip()

        query = re.sub(
            r"^\s*(can you\s+)?"
            r"(search|look up|google|find out|check online|"
            r"check the web|look online)"
            r"( this| that| it)?"
            r"( for me)?\s*[:,]?\s*",
            "",
            query,
            flags=re.IGNORECASE,
        )

        return query.strip()

    # =============================================================
    # Listening controls
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

    # =============================================================
    # Follow-up / correction detection
    # =============================================================

    @staticmethod
    def _is_correction(text: str) -> bool:
        q = text.lower().strip()

        patterns = (
            r"\bthat's wrong\b",
            r"\bthat is wrong\b",
            r"\bthis is wrong\b",
            r"\bthat's not right\b",
            r"\bthat is not right\b",
            r"\bthat's incorrect\b",
            r"\bthat is incorrect\b",
            r"\bnot true\b",
            r"\bthat's just not true\b",
            r"\bthat is just not true\b",
            r"\byou're wrong\b",
            r"\byou are wrong\b",
            r"\bno[, ]+that's wrong\b",
            r"\bno[, ]+that is wrong\b",
            r"\bwrong on so many levels\b",
            r"\b(all|every|those) (of )?(them|those|these) are wrong\b",
            r"\b(all|every) (of )?(them|those|these) are incorrect\b",
        )

        return any(
            re.search(pattern, q)
            for pattern in patterns
        )

    @staticmethod
    def _has_reference_language(text: str) -> bool:
        q = text.lower()

        return bool(
            re.search(
                r"\b("
                r"he|him|his|she|her|they|them|their|"
                r"it|that|this|those|these"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _has_continuation_phrase(text: str) -> bool:
        q = text.lower().strip()

        return bool(
            re.search(
                r"\b("
                r"elaborate"
                r"|expand on"
                r"|explain that"
                r"|tell me more"
                r"|more about that"
                r"|what about"
                r"|how about"
                r"|what happened next"
                r"|when did that happen"
                r"|why is that"
                r"|why did that"
                r"|and internationally"
                r"|and domestically"
                r"|and worldwide"
                r"|and globally"
                r")\b",
                q,
            )
        )

    @classmethod
    def _is_followup(cls, text: str) -> bool:
        q = text.lower().strip()

        if not q:
            return False

        # Pure arithmetic must NEVER become a search follow-up.
        if re.fullmatch(
            r"(what|how much|calculate|compute)\s+"
            r"[\d\s+\-*/().%]+[?!.\s]*",
            q,
        ):
            return False

        if cls._is_correction(text):
            return True

        if cls._has_reference_language(text):
            return True

        if cls._has_continuation_phrase(text):
            return True

        # Very short questions such as "what was it?" are obvious
        # continuations, but ordinary short questions are not.
        if len(q.split()) <= 5 and re.search(
            r"^(what|who|when|where|which|how|and)\b",
            q,
        ):
            return bool(
                re.search(
                    r"\b("
                    r"it|that|this|he|him|his|she|her|they|them|"
                    r"those|these"
                    r")\b",
                    q,
                )
            )

        return False

    def _followup_query(
        self,
        text: str,
    ) -> str | None:
        previous = self.search_state.query
        subject = self.search_state.subject

        if not previous:
            return None

        q = text.strip()

        # A correction means: redo the previous search.
        if self._is_correction(q):
            if subject:
                return (
                    f"{subject} latest "
                    f"information"
                )

            return previous

        # Explicitly resolve pronouns against the known subject.
        if subject and (
            self._has_reference_language(q)
            or self._has_continuation_phrase(q)
        ):
            return f"{subject} {q}"

        return previous

    # =============================================================
    # Current-information detection
    # =============================================================

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
                r"|currently"
                r"|today"
                r"|tonight"
                r"|right now"
                r"|now"
                r"|this week"
                r"|this month"
                r"|newest"
                r"|as of"
                r"|just released"
                r"|just announced"
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
                r"|check the web"
                r"|search it up"
                r"|search that"
                r"|look that up"
                r")\b",
                q,
            )
        )

    @staticmethod
    def _is_wikipedia_request(text: str) -> bool:
        return bool(
            re.search(
                r"\bwikipedia\b",
                text.lower(),
            )
        )

    @staticmethod
    def _is_source_request(text: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*("
                r"source|sources|citation|citations|"
                r"cite it|where did you get that|"
                r"where is that from|proof|link|links"
                r")\s*[?.!]*",
                text.lower(),
            )
        )

    # =============================================================
    # Relevance
    # =============================================================

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "the", "a", "an",
            "is", "are", "was", "were",
            "what", "when", "where", "who", "why", "how",
            "did", "does", "do",
            "has", "have", "had",
            "his", "her", "their", "its",
            "this", "that", "these", "those",
            "for", "from", "with", "about",
            "and", "or", "to", "of", "in", "on", "at", "by", "as",
            "it", "me", "you", "i", "he", "she", "they", "them",
            "be", "can", "could", "would", "should",
            "latest", "recent", "current",
            "right", "now",
            "please",
            "look", "up", "search", "online",
            "tell", "know", "think",
            "do", "you",
        }

        words = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        return {
            word
            for word in words
            if word not in stopwords
            and len(word) > 1
        }

    @classmethod
    def _result_relevance(
        cls,
        query: str,
        result: SearchResult,
    ) -> float:
        query_tokens = cls._tokens(query)

        if not query_tokens:
            return 0.0

        title_tokens = cls._tokens(
            result.title
        )

        body_tokens = cls._tokens(
            f"{result.title} {result.snippet}"
        )

        if not body_tokens:
            return 0.0

        body_overlap = (
            len(query_tokens & body_tokens)
            / len(query_tokens)
        )

        title_overlap = (
            len(query_tokens & title_tokens)
            / len(query_tokens)
        )

        return (
            body_overlap * 0.75
            + title_overlap * 0.25
        )

    @classmethod
    def _filter_relevant_results(
        cls,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        scored: list[
            tuple[float, SearchResult]
        ] = []

        for result in results:
            score = cls._result_relevance(
                query,
                result,
            )

            if score >= 0.18:
                scored.append(
                    (score, result)
                )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in scored[:6]
        ]

    # =============================================================
    # System prompt
    # =============================================================

    def _system(self) -> str:
        grounding = """
WEB EVIDENCE RULES — VERY IMPORTANT:

The application may provide WEB EVIDENCE.

The application performs searches. You do not personally browse the web.

For current-information questions:

- Use ONLY the supplied WEB EVIDENCE.
- Do not use pretrained knowledge to fill missing current facts.
- Never invent facts.
- Never invent citations.
- Never invent URLs.
- Never treat the user's correction or assertion as evidence.
- If the supplied evidence is insufficient, say that it cannot be verified.
- If sources disagree, explicitly say they disagree.
- Do not choose an unsupported answer simply because it sounds plausible.

SEARCH CORRECTIONS:

If the user says your previous answer was wrong, treat that as a request
to re-check the subject. The user's statement is not evidence.

SOURCE HONESTY:

Do not claim that you personally searched, browsed, opened, checked,
accessed, or visited a website.

The application appends the actual sources.

Do not generate a Sources section yourself.

STYLE:

You are Quinn, a wearable voice assistant.
Be calm, natural, concise, and direct.
Default to one to three short sentences.
Do not narrate internal reasoning.
"""

        return (
            f"{self.system_prompt}\n\n{grounding}"
        ).strip()

    # =============================================================
    # Evidence helpers
    # =============================================================

    @staticmethod
    def _evidence_from_dicts(
        results: list[dict],
    ) -> list[SearchResult]:
        converted: list[SearchResult] = []

        for item in results:
            if isinstance(
                item,
                SearchResult,
            ):
                converted.append(item)
                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            converted.append(
                SearchResult(
                    title=str(
                        item.get("title", "")
                    ),
                    url=str(
                        item.get("url", "")
                    ),
                    snippet=str(
                        item.get("snippet", "")
                    ),
                    source=str(
                        item.get("source", "Web")
                    ),
                    published=item.get(
                        "published"
                    ),
                    kind=str(
                        item.get("kind", "web")
                    ),
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

        chunks = [
            "WEB EVIDENCE:"
        ]

        for index, result in enumerate(
            results,
            start=1,
        ):
            chunks.append(
                "\n".join(
                    [
                        f"[{index}]",
                        f"Title: {result.title}",
                        f"Domain: {result.domain}",
                        f"Source type: {result.kind}",
                        f"Published: "
                        f"{result.published or 'unknown'}",
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
        seen: set[str] = set()
        lines: list[str] = []

        for result in results:
            if (
                not result.url
                or result.url in seen
            ):
                continue

            seen.add(result.url)

            label = (
                result.title
                or result.domain
                or "Source"
            )

            lines.append(
                f"- {label} ({result.domain})"
            )

            if len(lines) >= 5:
                break

        if not lines:
            return ""

        return (
            "\n\nSources:\n"
            + "\n".join(lines)
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

        if isinstance(
            web,
            dict,
        ):
            evidence = self._evidence_from_dicts(
                web.get("results", [])
            )

        elif isinstance(
            web,
            list,
        ):
            evidence = self._evidence_from_dicts(
                web
            )

        if web is not None and not evidence:
            return (
                "I couldn't verify that from "
                "the search results."
            )

        if evidence:
            prompt = (
                f"{self._format_evidence(evidence)}\n\n"
                f"USER QUESTION:\n{text}"
            )
        else:
            prompt = text

        messages: list[
            dict[str, str]
        ] = [
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
            temperature=(
                0.15
                if evidence
                else 0.4
            ),
        )

        answer = self._clean_response(
            concise(response.content)
        )

        self.context.add(
            "user",
            text,
        )

        self.context.add(
            "assistant",
            answer,
        )

        if evidence:
            answer += self._format_sources(
                evidence
            )

        return answer

    # =============================================================
    # Response cleanup
    # =============================================================

    @staticmethod
    def _clean_response(
        text: str,
    ) -> str:
        if not text:
            return (
                "I don't have an answer "
                "for that yet."
            )

        # Remove model-generated numeric citations.
        text = re.sub(
            r"\[\d+\]",
            "",
            text,
        )

        # Remove model-generated Sources sections.
        text = re.sub(
            r"\n+\s*Sources?\s*:.*$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        replacements = (
            (
                r"\bI (?:have )?"
                r"(?:searched|searched for|search)\b",
                "The search results show",
            ),
            (
                r"\bI (?:have )?"
                r"(?:browsed|looked up|looked online)\b",
                "The search results show",
            ),
            (
                r"\bI (?:have )?"
                r"(?:checked|accessed|visited) Wikipedia\b",
                "The supplied Wikipedia results show",
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

        if (
            not query
            or not self.settings.web_search_enabled
        ):
            return []

        if kind == "wikipedia":
            results = await self.search.wikipedia(
                query,
                limit=6,
            )

        elif kind == "news":
            results = await self.search.news(
                query,
                limit=8,
            )

            if len(results) < 3:
                extra = await self.search.search(
                    query,
                    limit=5,
                )

                existing_urls = {
                    result.url
                    for result in results
                }

                results.extend(
                    result
                    for result in extra
                    if result.url
                    not in existing_urls
                )

        else:
            results = await self.search.search(
                query,
                limit=8,
            )

        relevant = self._filter_relevant_results(
            query,
            results,
        )

        self._remember_search(
            query,
            relevant,
            kind,
        )

        log.info(
            "search_complete",
            extra={
                "query": query,
                "kind": kind,
                "raw_results": len(results),
                "relevant_results": len(relevant),
                "subject": self.search_state.subject,
            },
        )

        return relevant

    # =============================================================
    # Memory
    # =============================================================

    def _remember(
        self,
        text: str,
    ) -> None:
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
            self.memory.add(
                content
            )

    # =============================================================
    # Main handler
    # =============================================================

    async def handle(
        self,
        text: str,
    ) -> str:
        text = text.strip()

        if not text:
            return ""

        q = text.lower()

        route = self.router.classify(
            text
        )

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
        # Listening controls
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

            return (
                "Okay. I'll stop listening."
            )

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
                result = (
                    await self.emergency.activate()
                )

                if result.get("active"):
                    self.context.transition(
                        ConversationState.EMERGENCY
                    )

                    return (
                        "Emergency mode activated."
                    )

                return result.get(
                    "message",
                    "I couldn't activate emergency mode.",
                )

            if route.detail == "stop":
                result = (
                    await self.emergency.stop()
                )

                self.context.transition(
                    ConversationState.CONVERSATION
                )

                return result.get(
                    "message",
                    "Emergency mode stopped.",
                )

            return (
                "Do you want me to activate "
                "emergency mode?"
            )

        # ---------------------------------------------------------
        # Memory
        # ---------------------------------------------------------

        if route.route == Route.MEMORY_WRITE:
            try:
                self._remember(text)
                return "Got it."
            except Exception:
                log.exception(
                    "memory_write_failed"
                )
                return "I couldn't save that."

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

                return (
                    "I don't have that saved."
                )

            except Exception:
                log.exception(
                    "memory_delete_failed"
                )
                return (
                    "I couldn't update my memory."
                )

        if route.route == Route.MEMORY_READ:
            memories = self.memory.list()

            if not memories:
                return (
                    "I don't have any saved "
                    "memories yet."
                )

            lines = [
                "Here's what I remember:"
            ]

            for item in memories:
                content = (
                    item.get("content")
                    or item.get("text")
                    or str(item)
                )

                lines.append(
                    f"- {content}"
                )

            return "\n".join(lines)

        # ---------------------------------------------------------
        # Date
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
            now = (
                datetime.datetime
                .now()
                .astimezone()
            )

            return now.strftime(
                "Today is %A, %B %-d, %Y."
            )

        # ---------------------------------------------------------
        # Time
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"what time is it"
            r"|what(?:'s| is) the time"
            r"|current time"
            r")\b",
            q,
        ):
            now = (
                datetime.datetime
                .now()
                .astimezone()
            )

            return now.strftime(
                "It's %-I:%M %p."
            )

        # ---------------------------------------------------------
        # Source request
        # ---------------------------------------------------------

        if self._is_source_request(text):
            results = (
                self.search_state.results
                or []
            )

            if not results:
                return (
                    "I don't have a recent "
                    "search to cite."
                )

            return (
                "These are the sources from "
                "my most recent search:"
                + self._format_sources(
                    results
                )
            )

        # ---------------------------------------------------------
        # Search decision
        # ---------------------------------------------------------

        has_previous_search = bool(
            self.search_state.results
        )

        is_followup = (
            has_previous_search
            and self._is_followup(text)
        )

        # A correction is a special follow-up.
        # It means re-check the previous subject.
        if (
            is_followup
            and self._is_correction(text)
        ):
            followup_query = (
                self._followup_query(text)
            )

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

                return (
                    "I re-checked it, but I "
                    "couldn't find relevant "
                    "evidence to verify the answer."
                )

        needs_search = (
            route.route in (
                Route.SEARCH,
                Route.CURRENT_INFORMATION,
            )
            or self._is_explicit_search(text)
            or self._is_current_query(text)
            or self._is_wikipedia_request(text)
        )

        # ---------------------------------------------------------
        # Search-aware follow-up
        # ---------------------------------------------------------

        if (
            not needs_search
            and is_followup
        ):
            followup_query = (
                self._followup_query(text)
            )

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

                return (
                    "I couldn't find relevant "
                    "results to verify that."
                )

        # ---------------------------------------------------------
        # Normal / explicit search
        # ---------------------------------------------------------

        if needs_search:
            if self._is_wikipedia_request(text):
                kind = "wikipedia"
            elif getattr(
                route,
                "detail",
                None,
            ) == "news":
                kind = "news"
            else:
                kind = "web"

            # Construct a clean query.
            search_query = (
                self._clean_search_text(text)
            )

            web = await self._web_tool(
                search_query
            )

            raw_results = (
                self._evidence_from_dicts(
                    web.get("results", [])
                )
            )

            relevant = (
                self._filter_relevant_results(
                    search_query,
                    raw_results,
                )
            )

            subject = (
                self._extract_search_subject(
                    search_query,
                    relevant,
                )
            )

            self._remember_search(
                search_query,
                relevant,
                kind,
                subject,
            )

            if not relevant:
                return (
                    "I searched for that, but "
                    "I couldn't find relevant "
                    "results to verify it."
                )

            return await self._answer(
                text,
                web={
                    "ok": True,
                    "results": relevant,
                },
            )

        # ---------------------------------------------------------
        # Normal conversation
        # ---------------------------------------------------------

        return await self._answer(
            text
        )

    # =============================================================
    # Shutdown
    # =============================================================

    async def shutdown(self) -> None:
        try:
            if (
                getattr(
                    self.emergency,
                    "active",
                    False,
                )
            ):
                await self.emergency.stop()

        except Exception:
            log.exception(
                "emergency_shutdown_failed"
            )

    # =============================================================
    # Status
    # =============================================================

    async def status(
        self,
    ) -> dict[str, Any]:
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
            "last_search": (
                self.search_state.query
            ),
            "last_search_subject": (
                self.search_state.subject
            ),
            "last_search_results": len(
                self.search_state.results
                or []
            ),
        }