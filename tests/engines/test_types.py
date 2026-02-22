"""Tests for engine protocol and shared result types."""

from __future__ import annotations

import pytest

from grip.engines.types import AgentRunResult, EngineProtocol, ToolCallDetail

# -- ToolCallDetail ----------------------------------------------------------


class TestToolCallDetail:
    def test_creation_with_all_fields(self):
        """ToolCallDetail stores name, success, duration, and optional output preview."""
        detail = ToolCallDetail(
            name="read_file",
            success=True,
            duration_ms=42.5,
            output_preview="first 80 chars...",
        )
        assert detail.name == "read_file"
        assert detail.success is True
        assert detail.duration_ms == 42.5
        assert detail.output_preview == "first 80 chars..."

    def test_output_preview_defaults_to_empty(self):
        """output_preview should default to an empty string when omitted."""
        detail = ToolCallDetail(name="shell", success=False, duration_ms=100.0)
        assert detail.output_preview == ""


# -- AgentRunResult -----------------------------------------------------------


class TestAgentRunResult:
    def test_defaults(self):
        """Only response is required; every other field has a sensible default."""
        result = AgentRunResult(response="Hello")
        assert result.response == "Hello"
        assert result.iterations == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.tool_calls_made == []
        assert result.tool_details == []

    def test_total_tokens_property(self):
        """total_tokens is a computed property: prompt_tokens + completion_tokens."""
        result = AgentRunResult(
            response="done",
            prompt_tokens=150,
            completion_tokens=50,
        )
        assert result.total_tokens == 200

    def test_total_tokens_defaults_to_zero(self):
        """total_tokens is 0 when neither prompt nor completion tokens are set."""
        result = AgentRunResult(response="")
        assert result.total_tokens == 0

    def test_mutable_list_defaults_are_independent(self):
        """Each instance gets its own list objects, not shared mutable defaults."""
        a = AgentRunResult(response="a")
        b = AgentRunResult(response="b")
        a.tool_calls_made.append("x")
        assert b.tool_calls_made == []

    def test_tool_details_populated(self):
        """tool_details can hold a list of ToolCallDetail instances."""
        detail = ToolCallDetail(name="web_search", success=True, duration_ms=300.0)
        result = AgentRunResult(response="ok", tool_details=[detail])
        assert len(result.tool_details) == 1
        assert result.tool_details[0].name == "web_search"


# -- EngineProtocol -----------------------------------------------------------


class TestEngineProtocol:
    def test_cannot_instantiate_directly(self):
        """EngineProtocol is abstract and raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            EngineProtocol()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_created(self):
        """A subclass that implements all abstract methods can be instantiated."""

        class StubEngine(EngineProtocol):
            async def run(
                self,
                user_message: str,
                *,
                session_key: str = "cli:default",
                model: str | None = None,
            ) -> AgentRunResult:
                return AgentRunResult(response="stub")

            async def consolidate_session(self, session_key: str) -> None:
                pass

            async def reset_session(self, session_key: str) -> None:
                pass

        engine = StubEngine()
        assert isinstance(engine, EngineProtocol)

    def test_partial_subclass_still_raises(self):
        """A subclass missing one abstract method cannot be instantiated."""

        class PartialEngine(EngineProtocol):
            async def run(
                self,
                user_message: str,
                *,
                session_key: str = "cli:default",
                model: str | None = None,
            ) -> AgentRunResult:
                return AgentRunResult(response="partial")

            # consolidate_session intentionally omitted
            # reset_session intentionally omitted

        with pytest.raises(TypeError):
            PartialEngine()  # type: ignore[abstract]
