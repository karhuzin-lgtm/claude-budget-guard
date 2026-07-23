"""Detect a retry-loop: the same tool call repeating over and over.

The classic "agent stuck burning tokens" failure is a model that issues the
*identical* tool call again and again — re-running the same failing command,
re-reading the same file, retrying the same request — for minutes or hours as a
black box. Token/USD ceilings eventually catch this, but a loop can be caught
much earlier and much more cheaply by noticing the repetition itself.

We look at the *trailing* run of tool calls: how many times in a row, ending at
the most recent call, the exact same call has been made. Identity is the tool
name plus a canonical serialization of its input. When that run length reaches
the configured limit we report a loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

from .transcript import ToolCall


@dataclass
class LoopInfo:
    """Details of a detected trailing repeat run."""

    name: str
    count: int
    signature: str


def _signature(call: ToolCall) -> str:
    """Canonical identity of a tool call: name + sorted-key JSON of input."""
    try:
        payload = json.dumps(call.input, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str is broad
        payload = repr(call.input)
    return f"{call.name}\x00{payload}"


def trailing_repeat(calls: Sequence[ToolCall]) -> int:
    """Length of the run of identical calls ending at the most recent one."""
    if not calls:
        return 0
    last = _signature(calls[-1])
    count = 0
    for call in reversed(calls):
        if _signature(call) == last:
            count += 1
        else:
            break
    return count


def detect_loop(
    calls: Sequence[ToolCall], limit: Optional[int]
) -> Optional[LoopInfo]:
    """Return ``LoopInfo`` when the trailing identical run reaches ``limit``.

    ``limit`` of ``None`` or ``<= 0`` disables detection (returns ``None``).
    """
    if not limit or limit <= 0:
        return None
    if not calls:
        return None
    count = trailing_repeat(calls)
    if count >= limit:
        last = calls[-1]
        return LoopInfo(name=last.name, count=count, signature=_signature(last))
    return None


def loop_count(calls: Sequence[ToolCall]) -> int:
    """Convenience: the trailing identical-run length (0 for no calls)."""
    return trailing_repeat(list(calls))


__all__ = ["LoopInfo", "detect_loop", "trailing_repeat", "loop_count", "ToolCall"]
