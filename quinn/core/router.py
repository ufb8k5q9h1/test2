from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class Route(str, Enum):
    CONVERSATION = "conversation"
    CURRENT_INFORMATION = "current_information"
    SEARCH = "search"
    TOOL = "tool"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    MEMORY_DELETE = "memory_delete"
    SYSTEM_COMMAND = "system_command"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class RouteResult:
    route: Route
    explicit: bool = False
    detail: str = ""


class Router:
    """
    Deterministic first-pass intent router for Quinn.

    The router handles obvious intents before the LLM gets involved.
    Current-information requests are deliberately routed toward search.
    """

    current_terms = (
        "latest",
        "most recent",
        "recent",
        "recently",
        "current",
        "currently",
        "today",
        "tonight",
        "now",
        "right now",
        "this week",
        "this month",
        "this year",
        "as of",
        "newest",
        "updated",
        "just announced",
        "just released",
        "breaking",
        "ongoing",
    )

    news_terms = (
        "news",
        "headlines",
        "articles",
        "controversy",
        "controversial",
        "what happened",
        "what's happening",
        "whats happening",
        "passed away",
        "pass away",
        "died",
        "death",
        "dead",
        "obituary",
        "killed",
        "resigned",
        "arrested",
        "arrest",
        "accused",
        "accusations",
        "scandal",
        "incident",
        "breaking",
        "just announced",
    )

    release_terms = (
        "released",
        "release",
        "featured on",
        "featured in",
        "feature",
        "new song",
        "new album",
        "new movie",
        "new film",
        "latest song",
        "latest album",
        "latest movie",
        "latest film",
    )

    box_office_terms = (
        "box office",
        "gross",
        "grossed",
        "grossing",
        "worldwide gross",
        "domestic gross",
        "international gross",
        "internationally",
        "domestically",
    )

    office_terms = (
        "mayor",
        "president",
        "prime minister",
        "governor",
        "senator",
        "representative",
        "minister",
        "chancellor",
        "leader",
        "ceo",
        "chief executive",
    )

    search_verbs = (
        "search",
        "look up",
        "google",
        "find out",
        "check online",
        "check the web",
        "look online",
    )

    def classify(self, text: str) -> RouteResult:
        q = text.lower().strip()

        if not q:
            return RouteResult(Route.CONVERSATION)

        # ---------------------------------------------------------
        # System commands
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"stop listening"
            r"|go quiet"
            r"|cancel"
            r"|wait"
            r"|give me a minute"
            r"|give me a second"
            r"|hold on"
            r"|hang on"
            r"|don't answer yet"
            r"|do not answer yet"
            r"|keep listening"
            r"|continue listening"
            r"|resume listening"
            r")\b",
            q,
        ):
            return RouteResult(
                Route.SYSTEM_COMMAND,
                True,
            )

        # ---------------------------------------------------------
        # Emergency
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"stop emergency"
            r"|end emergency"
            r"|cancel emergency"
            r")\b",
            q,
        ):
            return RouteResult(
                Route.EMERGENCY,
                True,
                "stop",
            )

        if re.search(
            r"\b("
            r"i(?:'m| am) in an emergency"
            r"|activate emergency"
            r"|emergency mode"
            r"|start emergency"
            r")\b",
            q,
        ):
            return RouteResult(
                Route.EMERGENCY,
                True,
                "start",
            )

        if "emergency" in q:
            return RouteResult(
                Route.EMERGENCY,
                False,
                "ambiguous",
            )

        # ---------------------------------------------------------
        # Memory write
        # ---------------------------------------------------------

        if re.match(
            r"\s*(remember|don't forget|keep in mind)\b",
            q,
        ):
            return RouteResult(
                Route.MEMORY_WRITE,
                True,
            )

        # ---------------------------------------------------------
        # Memory delete
        # ---------------------------------------------------------

        if re.match(
            r"\s*forget\b",
            q,
        ):
            return RouteResult(
                Route.MEMORY_DELETE,
                True,
            )

        # ---------------------------------------------------------
        # Memory read
        #
        # Include natural-language variants. This MUST happen before
        # generic search/current-information detection.
        # ---------------------------------------------------------

        memory_read_patterns = (
            r"\bwhat do you remember\b",
            r"\bwhat do you know about me\b",
            r"\bwhat do you even know about me\b",
            r"\bhow much do you know about me\b",
            r"\bwhat have you remembered\b",
            r"\bwhat have you saved\b",
            r"\bwhat have you learned about me\b",
            r"\btell me what you remember\b",
            r"\btell me what you know about me\b",
            r"\bwhat memories do you have\b",
            r"\bwhat do you have saved about me\b",
            r"\bwhat is saved about me\b",
            r"\bmy memories\b",
            r"\bshow my memories\b",
            r"\blist my memories\b",
        )

        if any(
            re.search(pattern, q)
            for pattern in memory_read_patterns
        ):
            return RouteResult(
                Route.MEMORY_READ,
                True,
            )

        # ---------------------------------------------------------
        # Explicit search
        # ---------------------------------------------------------

        if re.search(
            r"\b("
            r"search"
            r"|look up"
            r"|google"
            r"|find out"
            r"|check online"
            r"|check the web"
            r"|look online"
            r")\b",
            q,
        ):
            if "wikipedia" in q:
                return RouteResult(
                    Route.SEARCH,
                    True,
                    "wikipedia",
                )

            if re.search(
                r"\b(news|headlines|articles)\b",
                q,
            ):
                return RouteResult(
                    Route.SEARCH,
                    True,
                    "news",
                )

            return RouteResult(
                Route.SEARCH,
                True,
                "web",
            )

        # ---------------------------------------------------------
        # Tools
        # ---------------------------------------------------------

        if re.match(
            r"\s*(text|message|tell)\s+\w+",
            q,
        ):
            return RouteResult(
                Route.TOOL,
                True,
                "message",
            )

        if re.search(
            r"\b("
            r"start recording"
            r"|begin recording"
            r"|stop recording"
            r"|end recording"
            r"|where am i"
            r"|get location"
            r"|my location"
            r")\b",
            q,
        ):
            return RouteResult(
                Route.TOOL,
                True,
            )

        # ---------------------------------------------------------
        # News / current information
        # ---------------------------------------------------------

        if any(
            term in q
            for term in self.news_terms
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "news",
            )

        if any(
            term in q
            for term in self.box_office_terms
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "web",
            )

        if any(
            term in q
            for term in self.release_terms
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "web",
            )

        if any(
            term in q
            for term in self.office_terms
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "web",
            )

        if any(
            term in q
            for term in self.current_terms
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "web",
            )

        # "last" is temporal too.
        if re.search(
            r"\b(last|latest|most recent)\b",
            q,
        ):
            return RouteResult(
                Route.CURRENT_INFORMATION,
                False,
                "web",
            )

        return RouteResult(
            Route.CONVERSATION
        )