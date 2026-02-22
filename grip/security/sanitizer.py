"""Secret detection and output sanitization.

Scans text for patterns that look like API keys, tokens, passwords,
and other secrets. Used to prevent accidental leakage in agent responses,
channel messages, and API outputs.
"""

from __future__ import annotations

import re

# Each pattern: (name, compiled regex, replacement)
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # API keys with common prefixes
    ("OpenAI API key", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    ("Anthropic API key", re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}")),
    ("Stripe key", re.compile(r"(sk|pk)_(test|live)_[a-zA-Z0-9]{20,}")),
    ("GitHub token", re.compile(r"(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}")),
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[a-zA-Z0-9_]{20,}")),
    ("Slack token", re.compile(r"xox[bpasr]-[a-zA-Z0-9\-]{20,}")),
    ("Slack webhook", re.compile(r"hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+")),
    ("Discord bot token", re.compile(r"[A-Za-z0-9]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}")),
    ("Telegram bot token", re.compile(r"\d{8,10}:[A-Za-z0-9_-]{35}")),
    ("AWS access key", re.compile(r"AKIA[A-Z0-9]{16}")),
    ("AWS secret key", re.compile(r"(?i)aws.{0,10}secret.{0,10}['\"][A-Za-z0-9/+=]{40}['\"]")),
    ("Google API key", re.compile(r"AIza[A-Za-z0-9_-]{35}")),
    ("Firebase key", re.compile(r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}")),
    ("Twilio key", re.compile(r"SK[a-f0-9]{32}")),
    ("Mailgun key", re.compile(r"key-[a-f0-9]{32}")),
    ("SendGrid key", re.compile(r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}")),
    ("Heroku API key", re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")),
    ("grip token", re.compile(r"grip_[A-Za-z0-9_-]{20,}")),
    # Generic patterns
    ("Bearer token in header", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*")),
    ("Private key block", re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    (
        "Connection string with password",
        re.compile(r"(?i)(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@"),
    ),
    (
        "Generic high-entropy secret",
        re.compile(
            r"(?i)(api[_-]?key|api[_-]?secret|auth[_-]?token|secret[_-]?key|"
            r"access[_-]?token|private[_-]?key|password)\s*[=:]\s*['\"]?[A-Za-z0-9+/=_\-]{16,}['\"]?"
        ),
    ),
]


def detect_secrets(text: str) -> list[tuple[str, str]]:
    """Scan text for potential secrets.

    Returns a list of (pattern_name, matched_text) tuples.
    """
    findings: list[tuple[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append((name, match.group()))
    return findings


def mask_secrets_in_text(text: str) -> str:
    """Replace detected secrets with masked versions.

    Preserves the first 4 and last 4 characters for identification,
    replaces the middle with asterisks.
    """
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(_mask_match, text)
    return text


def _mask_match(match: re.Match[str]) -> str:
    """Mask a matched secret, keeping prefix and suffix visible."""
    value = match.group()
    if len(value) <= 12:
        return value[:3] + "*" * (len(value) - 3)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]
