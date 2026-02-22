"""Email compose tool — tone-aware email drafting.

Composes email drafts (does NOT send). Tone presets adjust formatting:
formal, friendly, urgent, apologetic, followup. Saves drafts to the
workspace ``drafts/`` directory as ``.md`` files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from grip.tools.base import Tool, ToolContext

_TONE_TEMPLATES: dict[str, dict[str, str]] = {
    "formal": {
        "greeting": "Dear {recipient},",
        "closing": "Sincerely,\n{sender}",
        "style": "professional",
    },
    "friendly": {
        "greeting": "Hi {recipient},",
        "closing": "Best,\n{sender}",
        "style": "casual",
    },
    "urgent": {
        "greeting": "**[ACTION REQUIRED]**\n\nDear {recipient},",
        "closing": "This requires your immediate attention.\n\nRegards,\n{sender}",
        "style": "direct",
    },
    "apologetic": {
        "greeting": "Dear {recipient},\n\nI hope this message finds you well. I want to sincerely apologize",
        "closing": "I appreciate your understanding and patience.\n\nWith regards,\n{sender}",
        "style": "empathetic",
    },
    "followup": {
        "greeting": "Hi {recipient},\n\nFollowing up on our previous correspondence",
        "closing": "Looking forward to hearing from you.\n\nBest regards,\n{sender}",
        "style": "reference-prior",
    },
}


def _compose_email(
    tone: str,
    recipient: str,
    sender: str,
    subject: str,
    body: str,
    context: str = "",
) -> str:
    """Assemble an email draft with tone-appropriate formatting."""
    template = _TONE_TEMPLATES.get(tone, _TONE_TEMPLATES["formal"])

    greeting = template["greeting"].format(recipient=recipient, sender=sender)
    closing = template["closing"].format(recipient=recipient, sender=sender)

    parts = [f"**Subject:** {subject}", ""]

    if tone == "urgent":
        parts.append("**Priority:** HIGH")
        parts.append("")

    parts.append(greeting)
    parts.append("")

    if context and tone == "followup":
        parts.append(f"regarding {context},")
        parts.append("")

    if tone == "apologetic" and context:
        parts.append(f"for {context}.")
        parts.append("")

    parts.append(body)
    parts.append("")
    parts.append(closing)

    return "\n".join(parts)


class EmailComposeTool(Tool):
    """Tone-aware email drafting — composes but does NOT send."""

    @property
    def name(self) -> str:
        return "email_compose"

    @property
    def description(self) -> str:
        return "Compose a tone-aware email draft (formal/friendly/urgent/apologetic/followup). Does NOT send."

    @property
    def category(self) -> str:
        return "general"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tone": {
                    "type": "string",
                    "enum": ["formal", "friendly", "urgent", "apologetic", "followup"],
                    "description": "Email tone preset.",
                },
                "recipient": {
                    "type": "string",
                    "description": "Recipient name.",
                },
                "sender": {
                    "type": "string",
                    "description": "Sender name.",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "The main email body content.",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context (for followup: prior topic; for apologetic: what to apologize for).",
                },
            },
            "required": ["tone", "recipient", "sender", "subject", "body"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        tone = params.get("tone", "formal")
        recipient = params.get("recipient", "")
        sender = params.get("sender", "")
        subject = params.get("subject", "")
        body = params.get("body", "")
        context = params.get("context", "")

        if tone not in _TONE_TEMPLATES:
            return f"Error: unknown tone '{tone}'. Available: {', '.join(_TONE_TEMPLATES.keys())}"
        if not recipient:
            return "Error: recipient is required."
        if not subject:
            return "Error: subject is required."
        if not body:
            return "Error: body is required."

        draft = _compose_email(tone, recipient, sender, subject, body, context)

        drafts_dir = ctx.workspace_path / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_subject = (
            "".join(c if c.isalnum() or c in " -_" else "" for c in subject)[:50]
            .strip()
            .replace(" ", "_")
        )
        filename = f"{timestamp}_{safe_subject}.md"
        draft_path = drafts_dir / filename
        draft_path.write_text(draft, encoding="utf-8")

        return f"Draft saved to drafts/{filename}\n\n---\n\n{draft}"


def create_email_compose_tools() -> list[Tool]:
    """Factory function returning email compose tool instances."""
    return [EmailComposeTool()]
