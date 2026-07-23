"""Hook entry point: read stdin, decide, emit allow/warn/block.

Claude Code invokes this as a ``PreToolUse`` hook, passing a JSON object on
stdin::

    {"session_id": "...", "transcript_path": "/abs/session.jsonl",
     "cwd": "...", "hook_event_name": "PreToolUse",
     "tool_name": "Bash", "tool_input": {...}}

Contract:
    * allow  -> exit 0 (stdout ignored)
    * block  -> exit 2 with the reason on stderr (Claude Code shows it to the
                agent). Alternatively, with CLAUDE_BUDGET_OUTPUT=json, emit the
                official PreToolUse deny payload on stdout and exit 0::

                    {"hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "..."}}

                (The deprecated top-level ``{"decision":"block"}`` form is NOT
                used — PreToolUse may ignore it and let the call through.)

**Fail-open is absolute.** Every failure mode here — no stdin, bad JSON, missing
or unreadable transcript, parse error, config error — results in exit 0 / allow.
A guardrail must never be the reason a healthy session dies. The one and only
way to reach exit 2 is a *confirmed* over-budget or *confirmed* loop. A parse
that hits a size bound is flagged *partial*: a confirmed breach on the read
portion still blocks, but an unprovable one allows with a stderr warning.
"""

from __future__ import annotations

import json
import sys
from typing import IO, List, Optional, Tuple

from .budget import BLOCK, WARN, Decision, decide
from .config import Config, load_config
from .loops import detect_loop_run
from .transcript import (
    SessionUsage,
    ToolCall,
    TrailingRun,
    stream_session,
)

EXIT_ALLOW = 0
EXIT_BLOCK = 2

# Cap the PreToolUse stdin payload we read, in RAW BYTES. A tool_input is
# normally tiny; an unbounded read on a pathologically large payload could OOM
# the hook — bypassing the guard on exactly the big calls it should watch. Over
# the cap we fail-open (allow). 4 MiB is far above any real payload. We read from
# the byte buffer (not the decoded text stream) so the bound is on bytes, not
# Unicode code points (multibyte chars could otherwise blow past a char cap).
MAX_STDIN_BYTES = 4 * 1024 * 1024


def _read_stdin(stdin: IO[str]) -> Optional[dict]:
    try:
        # Prefer the raw BYTE buffer so the cap is enforced on bytes, not decoded
        # code points (real sys.stdin has .buffer). Fall back to the text stream
        # (e.g. an io.StringIO in tests). Read at most cap + 1 to detect oversize
        # without buffering the whole payload.
        buffer = getattr(stdin, "buffer", None)
        if buffer is not None:
            raw_bytes = buffer.read(MAX_STDIN_BYTES + 1)
            if raw_bytes is not None and len(raw_bytes) > MAX_STDIN_BYTES:
                return None  # oversized -> fail-open allow
            raw = raw_bytes.decode("utf-8", "replace") if raw_bytes else ""
        else:
            raw = stdin.read(MAX_STDIN_BYTES + 1)
            if raw is not None and len(raw) > MAX_STDIN_BYTES:
                return None  # oversized -> fail-open allow
    except (OSError, ValueError):
        return None
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _stream(transcript_path: Optional[str]) -> Tuple[SessionUsage, TrailingRun, bool]:
    """Single bounded pass over the transcript; fail-open to empty on error.

    Returns ``(SessionUsage, TrailingRun, partial)``. When the transcript could
    NOT be read — a missing/invalid path, or an ``OSError`` opening it — the read
    FAILED: it was never confirmed to be a complete, empty transcript. Token/USD
    accounting on a truly empty read is 0 and can never breach a ceiling, but the
    trailing-run SUFFIX is UNKNOWN, so the returned ``TrailingRun`` is marked
    ``tail_suffix_unknown`` (case (c)). Without this, an unavailable transcript
    produced a fresh trustworthy run; after folding the current PreToolUse call the
    trailing count became 1, and ``CLAUDE_BUDGET_LOOP_LIMIT=1`` would BLOCK the
    very first call — violating the documented unconditional fail-open on an
    unavailable transcript. ``partial`` stays ``False`` (nothing was truncated);
    only the loop path is gated, via ``tail_suffix_unknown``.

    A SUCCESSFUL complete read returns exactly as before — a trustworthy run whose
    loop may legitimately block.
    """
    if not transcript_path or not isinstance(transcript_path, str):
        return SessionUsage(), TrailingRun(tail_suffix_unknown=True), False
    try:
        return stream_session(transcript_path)
    except OSError:
        return SessionUsage(), TrailingRun(tail_suffix_unknown=True), False


def _append_current_call(run: TrailingRun, payload: dict) -> bool:
    """Fold the PreToolUse call *under evaluation* into the trailing run.

    PreToolUse fires BEFORE the call is written to the transcript, so the call
    being evaluated is not yet in the streamed state. Appending it here makes the
    Nth identical call (counting the current one) the one that blocks — correct
    pre-execution semantics — and prevents a DIFFERENT call from inheriting a
    stale N-repeat tail. Invalid shapes are skipped, never raised.

    Returns ``True`` when a valid current call was appended, ``False`` when the
    payload's ``tool_name``/``tool_input`` shape is invalid and nothing was
    folded. The caller MUST NOT let a loop block when this returns ``False``: the
    stale transcript tail may already be at the limit, but with no proven,
    identical current continuation a block would violate absolute fail-open
    (blocking a call never shown to match the loop).
    """
    name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if isinstance(name, str) and name and isinstance(tool_input, dict):
        run.add(ToolCall(name=name, input=tool_input))
        return True
    return False


def evaluate(payload: dict, config: Config) -> Tuple[Decision, bool]:
    """Pure-ish evaluation: given the hook payload and config, decide.

    Returns ``(Decision, partial)`` where ``partial`` is True when the transcript
    could not be fully accounted (a streaming bound tripped). Reads the transcript
    referenced by the payload but has no side effects on process state, which
    keeps it directly unit-testable.

    The token ceiling is checked INDEPENDENTLY of USD: the token total is derived
    first, then USD is computed in its own guarded step (an un-priceable overflow
    reads as +inf, never as an error). This ordering guarantees a pricing failure
    can never cancel an already-confirmed token-limit breach.
    """
    session, run, partial = _stream(payload.get("transcript_path"))

    # 1) Tokens — arbitrary-precision int sum, cannot overflow; always available.
    tokens = session.total_tokens

    # 2) USD — best-effort, and only when a USD ceiling is actually configured.
    #    Any pricing failure degrades to +inf ("over any finite ceiling"), never
    #    a raised exception that the outer fail-open would turn into an ALLOW.
    usd = 0.0
    if config.max_usd is not None:
        try:
            usd = config.pricing.session_cost(session)
        except (OverflowError, ValueError):
            usd = float("inf")

    # 3) Loop — the current PreToolUse call participates in the trailing run.
    #    A loop may only BLOCK when the trailing-run TAIL is fully known:
    #    ``not run.tail_suffix_unknown`` (no truncation, no read error, transcript
    #    available). This is INDEPENDENT of ``partial``: a usage/dedup/model bound
    #    can leave the token total incomplete (``partial``) while never dropping a
    #    ``tool_use`` block, so the trailing run is still fully observed and a real
    #    loop may block. Conversely a mid-stream malformed line breaks the run
    #    (resetting the counter) but, once a fresh run forms after it and reaches
    #    EOF, the tail is known and the loop may block. Only a genuinely unknown
    #    suffix leaves the loop UNPROVEN (warn, never block); a token/USD breach
    #    still blocks under a partial read (those metrics are monotonic).
    #
    #    A loop may additionally block ONLY when the current PreToolUse call was
    #    validly appended (``appended``): an invalid current call leaves the run
    #    at the OLD transcript tail, which could already be at the limit — but
    #    that tail is a different, unproven call, so blocking it would break
    #    fail-open. An unappended current call therefore forces the loop UNPROVEN
    #    (warn/allow at most), exactly like an unknown-suffix read. Appending a
    #    valid current call does NOT by itself restore trust: if the suffix is
    #    unknown the loop stays unproven regardless.
    loop = None
    loop_trustworthy = True
    if config.loop_limit:
        appended = _append_current_call(run, payload)
        loop = detect_loop_run(run, config.loop_limit)
        loop_trustworthy = (not run.tail_suffix_unknown) and appended

    return (
        decide(
            tokens=tokens,
            usd=usd,
            config=config,
            loop=loop,
            loop_trustworthy=loop_trustworthy,
        ),
        partial,
    )


def _safe_write(stream: IO[str], text: str) -> bool:
    """Best-effort write+flush that never raises. True iff the bytes are delivered.

    A ``write()`` alone is NOT proof of delivery: for a buffered stream (e.g.
    ``sys.stdout``) a successful ``write`` only fills the user-space buffer, and a
    later ``flush`` — or the interpreter-exit flush — can raise ``OSError`` AFTER
    this returned, so a deny packet reported "written" never reaches the pipe. We
    therefore flush SYNCHRONOUSLY here and fold any flush failure into the return,
    and treat a SHORT write (fewer characters accepted than we handed in, where
    the stream reports a count) as failure too. A closed/broken output stream — or
    any unexpected write/flush error — must never propagate: for a WARNING/ALLOW
    that keeps us fail-open, and for a confirmed BLOCK it lets emission fall back
    to the guaranteed exit-2 floor instead of the error tearing down the process
    (FIX 2).
    """
    try:
        written = stream.write(text)
        # Text streams return the number of characters written; a short write
        # means the tail never landed -> treat as a delivery failure. Streams that
        # return ``None`` give no count; nothing more we can check there.
        if written is not None and written < len(text):
            return False
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()
        return True
    except Exception:  # noqa: BLE001 - output must never break the guard
        return False


def _emit(decision: Decision, config: Config, stdout: IO[str], stderr: IO[str]) -> int:
    """Emit the decision and return the exit code. NEVER raises.

    A CONFIRMED block is hardened (FIX 2): its exit code is exit 2 regardless of
    whether writing the reason succeeded. In exit mode the stderr write+flush is
    best-effort and the code is always ``EXIT_BLOCK``. In JSON mode the deny
    payload is written AND flushed synchronously (via ``_safe_write``); if EITHER
    the write or the flush fails (closed/broken pipe, buffered flush error), we do
    NOT downgrade to allow — we fall back to exit 2 with a best-effort stderr
    reason. A WARNING or ALLOW stays fail-open (exit 0) even if its write fails;
    only confirmed-block emission is floored at exit 2.
    """
    if decision.action == BLOCK:
        reason = "[budget-guard] " + decision.message
        if config.output_mode == "json":
            # Official PreToolUse JSON contract (Claude Code hooks reference):
            # a deny travels in ``hookSpecificOutput`` with exit 0. The old
            # top-level ``{"decision":"block"}`` form is DEPRECATED for
            # PreToolUse and may be ignored — which would let the blocked call
            # through — so we never emit it.
            payload = json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
            if _safe_write(stdout, payload + "\n"):
                return EXIT_ALLOW
            # stdout is unusable: the deny payload never reached Claude Code, so
            # exit 0 would let the blocked call through. Fall back to the robust
            # exit-2 path (best-effort stderr reason) — a confirmed block must
            # never downgrade to allow because an output stream broke.
            _safe_write(stderr, reason + "\n")
            return EXIT_BLOCK
        # Exit mode: the exit code — not the stderr text — is what blocks, so the
        # write is best-effort and exit 2 is guaranteed either way.
        _safe_write(stderr, reason + "\n")
        return EXIT_BLOCK

    if decision.action == WARN:
        # Warnings never block; surface on stderr and allow (fail-open even if the
        # write fails).
        _safe_write(stderr, "[budget-guard] WARNING: " + decision.message + "\n")
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

        decision, partial = evaluate(payload, config)

        # Incomplete parse (a streaming bound tripped) with NO confirmed breach:
        # fail-open forbids blocking an unproven breach, but we must not hide the
        # incompleteness. Warn on stderr and allow. A CONFIRMED breach on the
        # partial data (decision blocked) is still a real breach -> fall through
        # to _emit and block.
        if partial and not decision.blocked:
            _safe_write(
                stderr,
                "[budget-guard] WARNING: transcript too large to fully "
                "account (partial read) — allowing\n",
            )
            return EXIT_ALLOW

        # ``_emit`` never raises: a confirmed block returns exit 2 even if the
        # output stream errors (FIX 2), so the catch-all below cannot swallow a
        # confirmed block into an allow.
        return _emit(decision, config, stdout, stderr)
    except Exception:  # noqa: BLE001 - deliberate catch-all: fail-open guardrail
        # Never let the guard break a working session.
        return EXIT_ALLOW


def main(argv: Optional[List[str]] = None) -> int:
    """Console/`python -m budget_guard` entry point."""
    return run(sys.stdin, sys.stdout, sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
