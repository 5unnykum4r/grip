"""Web tools: search the web and fetch page content.

web_search supports multiple backends (Brave, DuckDuckGo) with automatic
fallback. web_fetch retrieves a URL and extracts the main readable content.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import httpx
from loguru import logger

from grip.tools.base import Tool, ToolContext

_FETCH_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=5.0, pool=5.0)
_FETCH_MAX_CHARS = 50_000
_USER_AGENT = "grip/0.1 (AI Agent; +https://github.com/grip)"


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that strips tags and extracts readable content."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse multiple blank lines into one
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _extract_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


class WebSearchTool(Tool):
    @property
    def category(self) -> str:
        return "web"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and return a list of results with titles, URLs, and snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        query = params["query"]
        max_results = min(params.get("max_results", 5), 10)

        # Try Brave first if API key is available
        brave_key = ctx.extra.get("brave_api_key", "")
        if brave_key:
            result = await self._search_brave(query, max_results, brave_key)
            if result:
                return result

        # Fallback to DuckDuckGo HTML scrape
        return await self._search_duckduckgo(query, max_results)

    async def _search_brave(self, query: str, max_results: int, api_key: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("web", {}).get("results", [])
            if not results:
                return ""

            lines: list[str] = [f"Search results for: {query}\n"]
            for i, r in enumerate(results[:max_results], 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                snippet = r.get("description", "No description")
                lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")

            return "\n".join(lines)
        except httpx.HTTPError as exc:
            logger.warning("Brave search failed, falling back: {}", exc)
            return ""

    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        try:
            async with httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                )
                resp.raise_for_status()
                html = resp.text

            results = self._parse_ddg_html(html, max_results)
            if not results:
                return f"No results found for: {query}"

            lines: list[str] = [f"Search results for: {query}\n"]
            for i, (title, url, snippet) in enumerate(results, 1):
                lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")

            return "\n".join(lines)
        except httpx.HTTPError as exc:
            return f"Error: Web search failed: {exc}"

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[tuple[str, str, str]]:
        """Extract search results from DuckDuckGo HTML response."""
        results: list[tuple[str, str, str]] = []
        # Match result links and snippets from the HTML page
        link_pattern = re.compile(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)>', re.DOTALL
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i in range(min(len(links), max_results)):
            url, raw_title = links[i]
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            if url and title:
                # DDG wraps URLs through a redirect — extract the actual URL
                actual = re.search(r"uddg=([^&]+)", url)
                if actual:
                    from urllib.parse import unquote

                    url = unquote(actual.group(1))
                results.append((title, url, snippet))

        return results


class WebFetchTool(Tool):
    @property
    def category(self) -> str:
        return "web"

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a URL and extract its main readable text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch.",
                },
            },
            "required": ["url"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        url = params["url"]
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"

        try:
            async with httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type or "application/xhtml" in content_type:
                text = _extract_text(resp.text)
            elif "text/" in content_type or "json" in content_type or "xml" in content_type:
                text = resp.text
            else:
                return f"Fetched {url}: binary content ({content_type}, {len(resp.content)} bytes)"

            if len(text) > _FETCH_MAX_CHARS:
                text = text[:_FETCH_MAX_CHARS] + f"\n\n[truncated at {_FETCH_MAX_CHARS} characters]"

            return f"Content from {url}:\n\n{text}" if text else f"Fetched {url}: empty response"

        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} fetching {url}"
        except httpx.TimeoutException:
            return f"Error: Timeout fetching {url}"
        except httpx.HTTPError as exc:
            return f"Error: Failed to fetch {url}: {exc}"


def create_web_tools() -> list[Tool]:
    return [WebSearchTool(), WebFetchTool()]
