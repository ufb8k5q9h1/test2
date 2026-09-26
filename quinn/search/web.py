from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
import asyncio, urllib.parse, urllib.request

@dataclass
class SearchResult:
    title: str; url: str; snippet: str; source: str="DuckDuckGo"; timestamp: str|None=None
    def as_dict(self): return asdict(self)
class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int=5) -> list[SearchResult]: ...
class _DuckParser(HTMLParser):
    def __init__(self): super().__init__(); self.results=[]; self._link=None; self._title=""; self._snippet=""; self._snippet_on=False
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs); cls=attrs.get("class","")
        if tag=="a" and "result__a" in cls: self._link=attrs.get("href",""); self._title=""
        if tag in {"a","div"} and "result__snippet" in cls: self._snippet_on=True; self._snippet=""
    def handle_data(self,data):
        if self._link is not None: self._title+=data
        if self._snippet_on: self._snippet+=data
    def handle_endtag(self,tag):
        if tag=="a" and self._link is not None:
            if self._title.strip(): self.results.append(SearchResult(self._title.strip(),self._link,self._snippet.strip()))
            self._link=None
        if tag in {"a","div"}: self._snippet_on=False
class DuckDuckGoSearch(WebSearchProvider):
    async def search(self, query: str, limit: int=5) -> list[SearchResult]:
        def fetch():
            url="https://html.duckduckgo.com/html/?q="+urllib.parse.quote_plus(query)
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 Quinn/0.1"})
            with urllib.request.urlopen(req,timeout=12) as r: return r.read().decode(errors="replace")
        html=await asyncio.to_thread(fetch); parser=_DuckParser(); parser.feed(html); return parser.results[:limit]
