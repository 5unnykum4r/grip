"""Tests for the email_compose tool."""

from __future__ import annotations

import pytest

from grip.tools.base import ToolContext
from grip.tools.email_compose import (
    _TONE_TEMPLATES,
    EmailComposeTool,
    _compose_email,
    create_email_compose_tools,
)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


class TestComposeEmail:
    def test_formal_has_dear_and_sincerely(self):
        draft = _compose_email("formal", "John", "Jane", "Meeting", "Let's meet tomorrow.")
        assert "Dear John" in draft
        assert "Sincerely" in draft

    def test_friendly_has_hi(self):
        draft = _compose_email("friendly", "John", "Jane", "Quick note", "Hey!")
        assert "Hi John" in draft
        assert "Best" in draft

    def test_urgent_has_action_required(self):
        draft = _compose_email("urgent", "Team", "Manager", "Deadline", "Due today.")
        assert "ACTION REQUIRED" in draft
        assert "Priority" in draft.upper() or "HIGH" in draft

    def test_apologetic_has_apologize(self):
        draft = _compose_email(
            "apologetic", "Client", "Support", "Issue", "We are fixing it.", context="the outage"
        )
        assert "apologize" in draft.lower()
        assert "outage" in draft

    def test_followup_references_prior(self):
        draft = _compose_email(
            "followup", "Partner", "Me", "Update", "Any news?", context="the proposal"
        )
        assert "Following up" in draft
        assert "proposal" in draft

    def test_all_tones_produce_distinct_output(self):
        drafts = set()
        for tone in _TONE_TEMPLATES:
            draft = _compose_email(tone, "R", "S", "Subj", "Body text")
            drafts.add(draft)
        assert len(drafts) == len(_TONE_TEMPLATES)


class TestEmailComposeTool:
    def test_factory_returns_tool(self):
        tools = create_email_compose_tools()
        assert len(tools) == 1
        assert tools[0].name == "email_compose"

    @pytest.mark.asyncio
    async def test_creates_draft_file(self, ctx):
        tool = EmailComposeTool()
        result = await tool.execute(
            {
                "tone": "formal",
                "recipient": "John",
                "sender": "Jane",
                "subject": "Test Email",
                "body": "This is a test.",
            },
            ctx,
        )
        assert "Draft saved" in result
        drafts = list((ctx.workspace_path / "drafts").glob("*.md"))
        assert len(drafts) == 1

    @pytest.mark.asyncio
    async def test_missing_recipient_returns_error(self, ctx):
        tool = EmailComposeTool()
        result = await tool.execute(
            {
                "tone": "formal",
                "recipient": "",
                "sender": "Jane",
                "subject": "Test",
                "body": "Test body",
            },
            ctx,
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_unknown_tone_returns_error(self, ctx):
        tool = EmailComposeTool()
        result = await tool.execute(
            {
                "tone": "sarcastic",
                "recipient": "John",
                "sender": "Jane",
                "subject": "Test",
                "body": "Test body",
            },
            ctx,
        )
        assert "Error" in result
