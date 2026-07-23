"""Parse a Claude Code session transcript (JSONL) into token usage.

Claude Code appends one JSON object per line to the session ``transcript_path``.
Each line has a ``"type"`` of ``"user"``, ``"assistant"`` or ``"system"``. Only
assistant lines carry a ``message.usage`` object with the token counts.

The single most important correctness detail lives here: **dedup by
requestId**. Claude Code writes ONE assistant line PER CONTENT BLOCK, and every
line from the same model response shares the same ``requestId`` and repeats the
*same* ``usage`` object. Summing every line inflates the token total 2-3x, so we
count each ``usage`` exactly once per unique request key (``requestId``, falling
back to ``message.id``; if neither is present the line is skipped rather than
risk double counting).

Parsing is deliberately defensive: a malformed line, a missing field or an
unexpected type is skipped, never fatal. The hook that calls us is fail-open,
and this module honours the same contract by returning whatever it could read.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Usage:
    """Aggregated token counts for one or more model responses.

    Cache-creation and cache-read tokens are tracked separately from plain
    input tokens because they are billed at different rates (see ``pricing``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """All input-side tokens: fresh input + cache writes + cache reads."""
        return (
            self.input_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Grand total of every token that moved through the model."""
        return self.total_input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens
            + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass
class SessionUsage:
    """Usage for a whole session, broken down per model.

    A session can mix models (e.g. an Opus main loop with Haiku sub-agents), and
    USD cost depends on the model, so we keep a per-model breakdown alongside the
    request count. ``total`` collapses it to a single ``Usage`` for token limits.
    """

    by_model: Dict[str, Usage] = field(default_factory=dict)
    requests: int = 0

    def add(self, model: str, usage: Usage) -> None:
        current = self.by_model.get(model)
        self.by_model[model] = usage if current is None else current + usage
        self.requests += 1

    @property
    def total(self) -> Usage:
        total = Usage()
        for usage in self.by_model.values():
            total = total + usage
        return total

    @property
    def total_tokens(self) -> int:
        return self.total.total_tokens


@dataclass
class ToolCall:
    """A single ``tool_use`` block emitted by the assistant."""

    name: str
    input: dict


def _usage_from_dict(raw: object) -> Optional[Usage]:
    """Build a ``Usage`` from a raw ``message.usage`` dict, defensively."""
    if not isinstance(raw, dict):
        return None

    def _int(key: str) -> int:
        # Clamp to >= 0. A corrupt or session-writable transcript line carrying a
        # NEGATIVE count would otherwise *lower* the session total and could push
        # a truly over-budget session back under the ceiling — a guardrail bypass.
        # Fail-open must never become fail-under-count, so negatives read as 0.
        value = raw.get(key, 0)
        # json.loads accepts Infinity/-Infinity by default; int(float("inf"))
        # raises OverflowError. A single such transcript line must not crash the
        # parser (which the hook would turn into a fail-open allow, bypassing the
        # ceiling), so we reject non-finite floats and catch OverflowError too.
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return parsed if parsed > 0 else 0

    return Usage(
        input_tokens=_int("input_tokens"),
        output_tokens=_int("output_tokens"),
        cache_creation_tokens=_int("cache_creation_input_tokens"),
        cache_read_tokens=_int("cache_read_input_tokens"),
    )


def _request_key(record: dict, message: dict) -> Optional[str]:
    """Dedup key for a response: requestId, else message.id, else None."""
    key = record.get("requestId") or record.get("request_id")
    if isinstance(key, str) and key:
        return key
    mid = message.get("id")
    if isinstance(mid, str) and mid:
        return mid
    return None


def iter_records(lines):
    """Yield parsed JSON objects from an iterable of text lines.

    Malformed or non-object lines are skipped silently. Accepts any iterable so
    callers can feed a file handle, a list, or a fixture string split by line.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            yield obj


def parse_records(records) -> SessionUsage:
    """Aggregate token usage from an iterable of transcript records.

    Applies the requestId dedup rule so each model response is counted once.
    """
    session = SessionUsage()
    seen = set()

    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        usage = _usage_from_dict(message.get("usage"))
        if usage is None:
            continue

        key = _request_key(record, message)
        if key is None:
            # No stable identity -> cannot dedup safely -> skip to avoid inflation.
            continue
        if key in seen:
            continue
        seen.add(key)

        model = message.get("model")
        if not isinstance(model, str) or not model:
            model = "unknown"
        session.add(model, usage)

    return session


def iter_tool_calls(records) -> List[ToolCall]:
    """Extract assistant ``tool_use`` blocks in chronological order.

    Used by loop detection. ``message.content`` is a list of blocks; each block
    with ``type == "tool_use"`` carries a ``name`` and ``input``. Non-list
    content, string content and unknown block shapes are ignored.
    """
    calls: List[ToolCall] = []
    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name:
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {}
            calls.append(ToolCall(name=name, input=tool_input))
    return calls


def _read_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def parse_transcript(path: str) -> SessionUsage:
    """Read a transcript file and return its deduplicated ``SessionUsage``.

    Raises ``OSError`` if the file cannot be opened; the caller (the hook) turns
    that into a fail-open allow.
    """
    return parse_records(iter_records(_read_lines(path)))


def load_tool_calls(path: str) -> List[ToolCall]:
    """Read a transcript file and return its ``tool_use`` blocks in order."""
    return iter_tool_calls(iter_records(_read_lines(path)))
