from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from ddgs import DDGS


log = logging.getLogger("quinn.search")


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published: str | None = None
    kind: str = "web"

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.removeprefix("www.")
        except Exception:
            return self.source

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "domain": self.domain,
            "published": self.published,
            "kind": self.kind,
        }


class DuckDuckGoSearch:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    # =============================================================
    # Public API
    # =============================================================

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._search_sync,
            query,
            limit,
        )

    async def news(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._news_sync,
            query,
            limit,
        )

    async def wikipedia(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._wikipedia_sync,
            query,
            limit,
        )

    # =============================================================
    # Normal web
    # =============================================================

    def _search_sync(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        query = query.strip()

        if not query:
            return []

        log.info(
            "web_search",
            extra={"query": query},
        )

        results: list[SearchResult] = []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=limit,
                    backend="auto",
                    region="us-en",
                    safesearch="moderate",
                )

                for item in raw_results:
                    result = self._parse_result(
                        item,
                        kind="web",
                    )

                    if result is not None:
                        results.append(result)

        except Exception:
            log.exception(
                "web_search_error",
                extra={"query": query},
            )
            raise

        log.info(
            "web_search_complete",
            extra={
                "query": query,
                "results": len(results),
            },
        )

        return results

    # =============================================================
    # News
    # =============================================================

    def _news_sync(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        query = query.strip()

        if not query:
            return []

        log.info(
            "news_search",
            extra={"query": query},
        )

        results: list[SearchResult] = []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = ddgs.news(
                    query,
                    max_results=limit,
                    region="us-en",
                    safesearch="moderate",
                )

                for item in raw_results:
                    result = self._parse_result(
                        item,
                        kind="news",
                    )

                    if result is not None:
                        results.append(result)

        except Exception:
            log.exception(
                "news_search_error",
                extra={"query": query},
            )
            raise

        log.info(
            "news_search_complete",
            extra={
                "query": query,
                "results": len(results),
            },
        )

        return results

    # =============================================================
    # Wikipedia
    # =============================================================

    def _wikipedia_sync(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        query = query.strip()

        if not query:
            return []

        log.info(
            "wikipedia_search",
            extra={"query": query},
        )

        results: list[SearchResult] = []

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=limit,
                    backend="wikipedia",
                    region="us-en",
                    safesearch="moderate",
                )

                for item in raw_results:
                    result = self._parse_result(
                        item,
                        kind="wikipedia",
                    )

                    if result is not None:
                        results.append(result)

        except Exception:
            log.exception(
                "wikipedia_search_error",
                extra={"query": query},
            )
            raise

        log.info(
            "wikipedia_search_complete",
            extra={
                "query": query,
                "results": len(results),
            },
        )

        return results

    # =============================================================
    # Parsing
    # =============================================================

    @staticmethod
    def _parse_result(
        item: dict,
        kind: str,
    ) -> SearchResult | None:
        title = str(
            item.get("title", "")
        ).strip()

        url = str(
            item.get("href", "")
        ).strip()

        snippet = str(
            item.get("body", "")
        ).strip()

        published = item.get("date")

        if published is not None:
            published = str(published).strip() or None

        if not title and not snippet:
            return None

        source = (
            "Wikipedia"
            if kind == "wikipedia"
            else (
                "News"
                if kind == "news"
                else "Web"
            )
        )

        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            source=source,
            published=published,
            kind=kind,
        )