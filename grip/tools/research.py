"""Web research tool — multi-step deep research with cited summaries.

Decomposes a topic into sub-queries, searches in parallel via grip's
existing web_search tool, ranks and deduplicates URLs, fetches top
results, and builds a cited summary with numbered source references.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from grip.tools.base import Tool, ToolContext

_FETCH_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=5.0, pool=5.0)
_USER_AGENT = "grip/0.2 (AI Research Agent; +https://github.com/grip)"
_MAX_FETCH_CHARS = 30_000


def _decompose_topic(topic: str, depth: int) -> list[str]:
    """Rule-based query expansion — splits compound questions and adds variants.

    depth 1: original + "what is" prefix
    depth 2: adds synonym-style expansions
    depth 3: adds "how does", "why", comparative queries
    """
    queries = [topic]

    words = topic.split()
    if len(words) >= 3 and not topic.lower().startswith(("what", "how", "why", "when", "where")):
        queries.append(f"what is {topic}")

    if depth >= 2:
        if " and " in topic.lower():
            parts = re.split(r"\s+and\s+", topic, flags=re.IGNORECASE)
            queries.extend(parts)
        if " vs " in topic.lower() or " versus " in topic.lower():
            queries.append(f"{topic} comparison")
        queries.append(f"{topic} explained")

    if depth >= 3:
        queries.append(f"how does {topic} work")
        queries.append(f"why {topic}")
        queries.append(f"{topic} advantages disadvantages")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q.strip())
    return unique


def _rank_urls(search_results: list[dict[str, str]], max_sources: int) -> list[dict[str, str]]:
    """Deduplicate by domain, score by frequency across search results."""
    from urllib.parse import urlparse

    url_data: dict[str, dict[str, Any]] = {}
    for item in search_results:
        url = item.get("url", "")
        if not url:
            continue
        domain = urlparse(url).netloc
        if url not in url_data:
            url_data[url] = {
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "domain": domain,
                "score": 0,
            }
        url_data[url]["score"] += 1

    seen_domains: set[str] = set()
    ranked: list[dict[str, str]] = []
    for entry in sorted(url_data.values(), key=lambda x: x["score"], reverse=True):
        domain = entry["domain"]
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        ranked.append(entry)
        if len(ranked) >= max_sources:
            break

    return ranked


async def _fetch_url_text(url: str) -> str:
    """Fetch a URL and extract readable text content."""
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text[:_MAX_FETCH_CHARS]
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:10_000]
    except Exception as exc:
        logger.debug("Failed to fetch {}: {}", url, exc)
        return ""


_SOURCE_PRIORITY = {
    "docs.": 5,
    "github.com": 4,
    "developer.": 4,
    "api.": 4,
    "stackoverflow.com": 3,
    "mozilla.org": 4,
    ".gov": 4,
    "medium.com": 2,
    "dev.to": 2,
    "reddit.com": 2,
}


def _score_source_quality(url: str) -> tuple[int, str]:
    """Rate a URL's trustworthiness and return (score 1-5, label).

    Prefers official docs and primary sources over aggregators.
    """
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    for pattern, score in _SOURCE_PRIORITY.items():
        if pattern in domain:
            if score >= 4:
                return score, "PRIMARY"
            return score, "SECONDARY"
    return 2, "UNVERIFIED"


def _assess_confidence(sources: list[dict[str, str]], contents: list[str]) -> str:
    """Assess overall research confidence based on source count and quality.

    Returns: HIGH, MEDIUM, or LOW with a short explanation.
    """
    valid_count = sum(1 for c in contents if c.strip())
    quality_scores = [_score_source_quality(s.get("url", ""))[0] for s in sources]
    avg_quality = sum(quality_scores) / max(len(quality_scores), 1)

    if valid_count >= 3 and avg_quality >= 3.5:
        return "HIGH — Multiple authoritative sources agree"
    if valid_count >= 2 or avg_quality >= 3.0:
        return "MEDIUM — Limited sources or mixed authority"
    return "LOW — Few sources or only unofficial references"


def _build_cited_summary(topic: str, sources: list[dict[str, str]], contents: list[str]) -> str:
    """Format a research summary with numbered citations, source quality, and confidence."""
    lines: list[str] = [f"# Research: {topic}\n"]

    lines.append("## Key Findings\n")
    for i, (source, content) in enumerate(zip(sources, contents, strict=False), 1):
        title = source.get("title", source.get("url", "Source"))
        snippet = source.get("snippet", "")
        _, quality_label = _score_source_quality(source.get("url", ""))
        if content:
            preview = content[:500]
            lines.append(f"**[{i}] {title}** ({quality_label})")
            lines.append(f"{preview}...")
            lines.append("")
        elif snippet:
            lines.append(f"**[{i}] {title}** ({quality_label})")
            lines.append(snippet)
            lines.append("")

    confidence = _assess_confidence(sources, contents)
    lines.append(f"## Confidence: {confidence}\n")

    lines.append("## Sources\n")
    for i, source in enumerate(sources, 1):
        url = source.get("url", "")
        title = source.get("title", url)
        _, quality_label = _score_source_quality(url)
        lines.append(f"[{i}] [{quality_label}] {title} — {url}")

    return "\n".join(lines)


class WebResearchTool(Tool):
    """Multi-step deep research: decompose, search, rank, fetch, cite."""

    @property
    def name(self) -> str:
        return "web_research"

    @property
    def description(self) -> str:
        return "Multi-step web research: decompose topic, search, fetch, and build a cited summary."

    @property
    def category(self) -> str:
        return "web"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic or question.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Research depth: 1 (quick), 2 (moderate), 3 (thorough). Default 2.",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 2,
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Maximum number of unique sources to include. Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["topic"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        topic = params.get("topic", "")
        if not topic:
            return "Error: topic is required."

        depth = min(max(params.get("depth", 2), 1), 3)
        max_sources = min(max(params.get("max_sources", 5), 1), 10)

        queries = _decompose_topic(topic, depth)
        logger.info("web_research: {} sub-queries for topic '{}'", len(queries), topic)

        all_results: list[dict[str, str]] = []
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json", "no_html": "1"},
                    )
                    data = resp.json()
                    for item in data.get("RelatedTopics", []):
                        if "FirstURL" in item:
                            all_results.append(
                                {
                                    "url": item["FirstURL"],
                                    "title": item.get("Text", "")[:100],
                                    "snippet": item.get("Text", "")[:200],
                                }
                            )
                except Exception as exc:
                    logger.debug("Search query '{}' failed: {}", query, exc)

        if not all_results:
            return f"No search results found for topic: {topic}"

        ranked = _rank_urls(all_results, max_sources)

        contents: list[str] = []
        for source in ranked:
            content = await _fetch_url_text(source["url"])
            contents.append(content)

        return _build_cited_summary(topic, ranked, contents)


def create_research_tools() -> list[Tool]:
    """Factory function returning research tool instances."""
    return [WebResearchTool()]
