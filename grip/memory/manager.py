"""Two-layer memory system: long-term facts + searchable history.

MEMORY.md stores extracted key facts and decisions that persist across
sessions. HISTORY.md is an append-only log of conversation summaries
that can be grepped for context retrieval.

Consolidation is triggered when a session grows past the memory window
threshold: older messages are summarized by the LLM, facts are extracted
into MEMORY.md, and a summary is appended to HISTORY.md.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from grip.providers.types import LLMMessage, LLMProvider


class MemoryManager:
    """Manages MEMORY.md and HISTORY.md within the workspace.

    Provides read/write for both files and an LLM-driven consolidation
    process that extracts durable facts from old conversation messages.
    """

    _HISTORY_MAX_BYTES: int = 512_000

    def __init__(self, workspace_path: Path) -> None:
        self._memory_dir = workspace_path / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_path = self._memory_dir / "MEMORY.md"
        self._history_path = self._memory_dir / "HISTORY.md"

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    @property
    def history_path(self) -> Path:
        return self._history_path

    def read_memory(self) -> str:
        """Read the full contents of MEMORY.md."""
        if self._memory_path.exists():
            return self._memory_path.read_text(encoding="utf-8")
        return ""

    def write_memory(self, content: str) -> None:
        """Overwrite MEMORY.md with new content (atomic write)."""
        tmp = self._memory_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(self._memory_path)

    def append_to_memory(self, entry: str) -> None:
        """Append a new fact or section to the end of MEMORY.md."""
        current = self.read_memory()
        if current and not current.endswith("\n"):
            current += "\n"
        self.write_memory(current + entry.rstrip() + "\n")

    def read_history(self) -> str:
        """Read the full contents of HISTORY.md."""
        if self._history_path.exists():
            return self._history_path.read_text(encoding="utf-8")
        return ""

    def search_history(
        self, query: str, *, max_results: int = 20, decay_rate: float = 0.001
    ) -> list[str]:
        """Search HISTORY.md using keyword-weighted relevance scoring with time decay.

        Tokenizes the query into keywords, scores each history line by
        TF-IDF-style relevance (term frequency * inverse document frequency),
        applies a time-decay factor so recent entries rank higher, and
        returns the top results sorted by score descending.
        Falls back to simple substring match if the query is very short.
        """
        content = self.read_history()
        if not content:
            return []

        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return []

        query_tokens = _tokenize(query)

        if len(query_tokens) <= 1:
            query_lower = query.lower()
            return [line for line in lines if query_lower in line.lower()][:max_results]

        doc_freq: Counter[str] = Counter()
        line_token_sets: list[set[str]] = []
        for line in lines:
            tokens = set(_tokenize(line))
            line_token_sets.append(tokens)
            for token in tokens:
                doc_freq[token] += 1

        total_docs = len(lines)
        now = datetime.now(UTC)

        scored: list[tuple[float, str]] = []
        for line in lines:
            line_tokens = _tokenize(line)
            if not line_tokens:
                continue
            tf_counts = Counter(line_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt in tf_counts:
                    tf = tf_counts[qt] / len(line_tokens)
                    df = doc_freq.get(qt, 0)
                    idf = math.log((total_docs + 1) / (df + 1)) + 1.0
                    score += tf * idf
            if score > 0 and decay_rate > 0:
                ts_match = _TIMESTAMP_RE.match(line)
                if ts_match:
                    try:
                        entry_time = datetime.strptime(
                            ts_match.group(1), "%Y-%m-%d %H:%M:%S"
                        ).replace(tzinfo=UTC)
                        age_hours = (now - entry_time).total_seconds() / 3600
                        score *= 1.0 / (1.0 + age_hours * decay_rate)
                    except ValueError:
                        pass
            if score > 0:
                scored.append((score, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line in scored[:max_results]]

    def search_memory(
        self, query: str, *, max_results: int = 10, category: str | None = None
    ) -> list[str]:
        """Search MEMORY.md using keyword-weighted relevance scoring.

        Same TF-IDF approach as search_history but applied to the structured
        facts in MEMORY.md. Returns matching bullet points or sections.
        When category is provided, only entries matching ``- [category]`` are searched.
        """
        content = self.read_memory()
        if not content:
            return []

        chunks: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                chunks.append(stripped)

        if category:
            chunks = [c for c in chunks if c.startswith(f"- [{category}]")]

        if not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return chunks[:max_results]

        # Simple substring for single-word queries
        if len(query_tokens) <= 1:
            query_lower = query.lower()
            return [c for c in chunks if query_lower in c.lower()][:max_results]

        # TF-IDF scoring
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            for token in set(_tokenize(chunk)):
                doc_freq[token] += 1

        total = len(chunks)
        scored: list[tuple[float, str]] = []
        for chunk in chunks:
            chunk_tokens = _tokenize(chunk)
            if not chunk_tokens:
                continue
            tf_counts = Counter(chunk_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt in tf_counts:
                    tf = tf_counts[qt] / len(chunk_tokens)
                    df = doc_freq.get(qt, 0)
                    idf = math.log((total + 1) / (df + 1)) + 1.0
                    score += tf * idf
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:max_results]]

    def get_memory_stats(self) -> dict[str, Any]:
        """Return statistics about memory and history usage."""
        content = self.read_memory()
        chunks = [line.strip() for line in content.splitlines() if line.strip()]
        categories: Counter[str] = Counter()
        for chunk in chunks:
            cat_match = re.match(r"^- \[(\w+)\]", chunk)
            if cat_match:
                categories[cat_match.group(1)] += 1
        history_bytes = 0
        if self._history_path.exists():
            history_bytes = self._history_path.stat().st_size
        return {
            "total_entries": len(chunks),
            "categories": dict(categories),
            "memory_size_bytes": len(content.encode("utf-8")),
            "history_size_bytes": history_bytes,
        }

    def compact_memory(self, similarity_threshold: float = 0.7) -> int:
        """Deduplicate memory entries using Jaccard similarity on token sets.

        Returns number of entries removed.
        """
        content = self.read_memory()
        if not content:
            return 0
        chunks = [line.strip() for line in content.splitlines() if line.strip()]
        if len(chunks) < 2:
            return 0

        token_sets = [set(_tokenize(chunk)) for chunk in chunks]
        keep = [True] * len(chunks)

        for i in range(len(chunks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(chunks)):
                if not keep[j] or not token_sets[i] or not token_sets[j]:
                    continue
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                if union > 0 and intersection / union >= similarity_threshold:
                    keep[j] = False

        deduplicated = [c for c, k in zip(chunks, keep, strict=True) if k]
        removed = len(chunks) - len(deduplicated)
        if removed > 0:
            self.write_memory("\n".join(deduplicated) + "\n")
            logger.info("Memory compacted: removed {} duplicate entries", removed)
        return removed

    def _rotate_history(self) -> None:
        """Archive older half of HISTORY.md when file exceeds size threshold."""
        lines = self._history_path.read_text(encoding="utf-8").splitlines(keepends=True)
        midpoint = len(lines) // 2
        if midpoint == 0:
            return
        archive_name = f"HISTORY.archive.{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.md"
        archive_path = self._memory_dir / archive_name
        tmp_archive = archive_path.with_suffix(".tmp")
        tmp_archive.write_text("".join(lines[:midpoint]), encoding="utf-8")
        tmp_archive.rename(archive_path)
        tmp_history = self._history_path.with_suffix(".tmp")
        tmp_history.write_text("".join(lines[midpoint:]), encoding="utf-8")
        tmp_history.rename(self._history_path)
        logger.info("Rotated HISTORY.md: archived {} lines to {}", midpoint, archive_name)

    def append_history(self, entry: str) -> None:
        """Append a timestamped entry to HISTORY.md."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{timestamp}] {entry.rstrip()}\n"

        with self._history_path.open("a", encoding="utf-8") as f:
            f.write(line)

        try:
            if self._history_path.stat().st_size > self._HISTORY_MAX_BYTES:
                self._rotate_history()
        except OSError:
            pass

    def needs_consolidation(self, message_count: int, memory_window: int) -> bool:
        """Check if consolidation should run based on message count vs window."""
        return message_count > memory_window * 2

    async def consolidate(
        self,
        old_messages: list[LLMMessage],
        provider: LLMProvider,
        model: str,
    ) -> str:
        """Extract key facts from old messages using the LLM.

        Sends the old messages to the LLM with a consolidation prompt,
        appends extracted facts to MEMORY.md, and writes a summary
        to HISTORY.md.

        Returns the extracted facts string.
        """
        if not old_messages:
            return ""

        conversation_text = self._format_messages_for_consolidation(old_messages)

        consolidation_prompt = (
            "You are a memory consolidation assistant. Review the following conversation "
            "and extract the key facts, decisions, and important information that should "
            "be remembered long-term.\n\n"
            "Rules:\n"
            "- Extract only durable facts (user preferences, project decisions, names, "
            "technical choices, important outcomes).\n"
            "- Skip transient information (greetings, small talk, tool execution details).\n"
            "- Format as a bulleted list with concise entries.\n"
            "- If there are no important facts to extract, respond with 'No new facts.'\n\n"
            f"Conversation:\n{conversation_text}"
        )

        logger.info("Running memory consolidation on {} messages", len(old_messages))

        response = await provider.chat(
            [
                LLMMessage(role="system", content="You extract key facts from conversations."),
                LLMMessage(role="user", content=consolidation_prompt),
            ],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )

        facts = response.content or ""

        if facts and "no new facts" not in facts.lower():
            self.append_to_memory(
                f"\n### Consolidated {datetime.now(UTC).strftime('%Y-%m-%d')}\n{facts}\n"
            )
            logger.info("Appended consolidated facts to MEMORY.md")

        summary = self._build_history_summary(old_messages)
        self.append_history(summary)
        logger.info("Appended summary to HISTORY.md")

        return facts

    @staticmethod
    def _format_messages_for_consolidation(messages: list[LLMMessage]) -> str:
        """Convert messages to a readable text block for the consolidation LLM."""
        lines: list[str] = []
        for msg in messages:
            if msg.role == "system":
                continue
            prefix = msg.role.upper()
            if msg.content:
                content = msg.content[:2000]
                lines.append(f"{prefix}: {content}")
            if msg.tool_calls:
                tool_names = ", ".join(tc.function_name for tc in msg.tool_calls)
                lines.append(f"{prefix}: [called tools: {tool_names}]")
        return "\n".join(lines)

    @staticmethod
    def _build_history_summary(messages: list[LLMMessage]) -> str:
        """Build a one-line summary of a batch of messages for HISTORY.md."""
        user_msgs = [m for m in messages if m.role == "user" and m.content]
        if not user_msgs:
            return f"Consolidated {len(messages)} messages (no user content)"

        topics: list[str] = []
        for m in user_msgs[:5]:
            snippet = (m.content or "")[:80].replace("\n", " ").strip()
            if snippet:
                topics.append(snippet)

        return f"Consolidated {len(messages)} messages. Topics: {'; '.join(topics)}"


# Stopwords filtered out during tokenization to improve TF-IDF relevance
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "but",
        "not",
        "with",
        "from",
        "by",
        "as",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "all",
        "each",
        "every",
        "some",
        "any",
        "no",
        "just",
        "about",
        "up",
        "out",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "also",
        "here",
        "there",
    }
)

_WORD_RE = re.compile(r"[a-z0-9_]+")
_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\]")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, filtering stopwords and short tokens."""
    return [
        word for word in _WORD_RE.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS
    ]


def build_memory_tools_description() -> dict[str, Any]:
    """Return a tool definition that the agent can use to search memory.

    This is registered as an additional tool so the agent can proactively
    search its own history when it needs to recall something.
    """
    return {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the conversation history log for past interactions matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find in conversation history.",
                    },
                },
                "required": ["query"],
            },
        },
    }
