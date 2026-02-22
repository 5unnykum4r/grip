"""Tests for the web_research tool."""

from __future__ import annotations

import pytest

from grip.tools.base import ToolContext
from grip.tools.research import (
    WebResearchTool,
    _build_cited_summary,
    _decompose_topic,
    _rank_urls,
    create_research_tools,
)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


class TestDecomposeTopic:
    def test_depth_1_adds_what_is(self):
        queries = _decompose_topic("machine learning algorithms", 1)
        assert "machine learning algorithms" in queries
        assert any("what is" in q.lower() for q in queries)

    def test_depth_1_skips_what_is_for_questions(self):
        queries = _decompose_topic("what is Python", 1)
        assert len([q for q in queries if q.lower().startswith("what is")]) == 1

    def test_depth_2_splits_and(self):
        queries = _decompose_topic("cats and dogs", 2)
        assert "cats" in queries
        assert "dogs" in queries

    def test_depth_3_adds_how_does(self):
        queries = _decompose_topic("neural networks", 3)
        assert any("how does" in q.lower() for q in queries)

    def test_deduplicates_queries(self):
        queries = _decompose_topic("simple", 3)
        assert len(queries) == len(set(q.lower() for q in queries))


class TestRankUrls:
    def test_deduplicates_by_domain(self):
        results = [
            {"url": "https://example.com/page1", "title": "A", "snippet": "a"},
            {"url": "https://example.com/page2", "title": "B", "snippet": "b"},
            {"url": "https://other.com/page1", "title": "C", "snippet": "c"},
        ]
        ranked = _rank_urls(results, 5)
        domains = [r["domain"] for r in ranked]
        assert len(domains) == len(set(domains))

    def test_respects_max_sources(self):
        results = [
            {"url": f"https://site{i}.com/page", "title": f"S{i}", "snippet": "x"}
            for i in range(10)
        ]
        ranked = _rank_urls(results, 3)
        assert len(ranked) <= 3


class TestBuildCitedSummary:
    def test_includes_citations(self):
        sources = [{"url": "https://a.com", "title": "Source A", "snippet": "info"}]
        contents = ["Some content about the topic"]
        result = _build_cited_summary("test topic", sources, contents)
        assert "[1]" in result
        assert "Source A" in result
        assert "https://a.com" in result

    def test_empty_content_uses_snippet(self):
        sources = [{"url": "https://b.com", "title": "Source B", "snippet": "fallback info"}]
        contents = [""]
        result = _build_cited_summary("topic", sources, contents)
        assert "fallback info" in result


class TestWebResearchTool:
    def test_factory_returns_tool(self):
        tools = create_research_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_research"

    def test_tool_properties(self):
        tool = WebResearchTool()
        assert tool.category == "web"
        assert "topic" in tool.parameters["properties"]
        assert "topic" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_empty_topic_returns_error(self, ctx):
        tool = WebResearchTool()
        result = await tool.execute({"topic": ""}, ctx)
        assert "Error" in result
