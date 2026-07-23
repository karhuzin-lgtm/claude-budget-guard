"""Hook entry point: read stdin, decide, emit allow/warn/block.

Claude Code invokes this as a ``PreToolUse`` hook, passing a JSON object on
stdin::

    {"session_id": "...", "transcript_path": "/abs/session.jsonl",
     "cwd": "...", "hook_event_name": "PreToolUse",
     "tool_name": "Bash", "tool_input": {...}}

Contract:
    * allow  -> exit 0 (stdout ignored)
    * block  -> exit 2 with the reason on stderr (Claude Code shows it to the
                agent). Alternatively, with CLAUDE_BUDGET_OUTPUT=json, emit
                ``{"decision":"block","reason":"..."}`` on stdout and exit 0.

**Fail-open is absolute.** Every failure mode here — no stdin, bad JSON, missing
or unreadable transcript, parse error, config error — results in exit 0 / allow.
A guardrail must never be the reason a healthy session dies. The one and only
way to reach exit 2 is a *confirmed* over-budget or *confirmed* loop.
"""

from __future__ import annotations

import json
import sys
from typing import IO, List, Optional

from .budget import BLOCK, WARN, Decision, decide
from .config import Config, load_config
from .loops import detect_loop
from .transcript import (
    iter_records,
    iter_tool_calls,
    parse_records,
)

EXIT_ALLOW = 0
EXIT_BLOCK = 2


def _read_stdin(stdin: IO[str]) -> Optional[dict]:
    try:
        raw = stdin.read()
    except (OSError, ValueError):
        return None
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_records(transcript_path: Optional[str]) -> List[dict]:
    if not transcript_path or not isinstance(transcript_path, str):
        return []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as fh:
            return list(iter_records(fh))
    except OSError:
        return []


def evaluate(payload: dict, config: Config) -> Decision:
    """Pure-ish evaluation: given the hook payload and config, decide.

    Reads the transcript referenced by the payload but has no side effects on
    process state, which keeps it directly unit-testable.
    """
    records = _read_records(payload.get("transcript_path"))

    session = parse_records(records)
    tokens = session.total_tokens
    usd = config.pricing.session_cost(session) if config.max_usd is not None else 0.0

    loop = None
    if config.loop_limit:
        calls = iter_tool_calls(records)
        loop = detect_loop(calls, config.loop_limit)

    return decide(tokens=tokens, usd=usd, config=config, loop=loop)


def _emit(decision: Decision, config: Config, stdout: IO[str], stderr: IO[str]) -> int:
    if decision.action == BLOCK:
        reason = "[budget-guard] " + decision.message
        if config.output_mode == "json":
            stdout.write(json.dumps({"decision": "block", "reason": reason}))
            stdout.write("\n")
            return EXIT_ALLOW
        stderr.write(reason + "\n")
        return EXIT_BLOCK

    if decision.action == WARN:
        # Warnings never block; surface on stderr and allow.
        stderr.write("[budget-guard] WARNING: " + decision.message + "\n")
        return EXIT_ALLOW

    return EXIT_ALLOW


def run(
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Full hook run over the given streams. Always returns an exit code.

    Wrapped in a broad try/except so that *any* unforeseen error is fail-open.
    """
    try:
        payload = _read_stdin(stdin)
        if payload is None:
            return EXIT_ALLOW

        config = load_config()

        # Zero-config: nothing to enforce -> allow. Never surprise-block.
        if not config.has_any_limit:
            return EXIT_ALLOW

        decision = evaluate(payload, config)
        return _emit(decision, config, stdout, stderr)
    except Exception:  # noqa: BLE001 - deliberate catch-all: fail-open guardrail
        # Never let the guard break a working session.
        return EXIT_ALLOW


def main(argv: Optional[List[str]] = None) -> int:
    """Console/`python -m budget_guard` entry point."""
    return run(sys.stdin, sys.stdout, sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
