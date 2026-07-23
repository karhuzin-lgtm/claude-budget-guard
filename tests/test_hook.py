"""End-to-end hook tests: fake stdin + fixture transcript -> exit code.

Exercises the full ``run()`` path including config-from-env, transcript reading,
decision and emission, plus the fail-open guarantees.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import contextmanager

from budget_guard.hook import EXIT_ALLOW, EXIT_BLOCK, run

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DEDUP = os.path.join(FIXTURES, "dedup.jsonl")
LOOP = os.path.join(FIXTURES, "loop.jsonl")
LOOP3 = os.path.join(FIXTURES, "loop3.jsonl")
EMPTY = os.path.join(FIXTURES, "empty.jsonl")

# Every CLAUDE_BUDGET_* var the hook reads; cleared before each run so tests are
# isolated from the ambient environment.
_ENV_KEYS = [
    "CLAUDE_BUDGET_MAX_TOKENS",
    "CLAUDE_BUDGET_MAX_USD",
    "CLAUDE_BUDGET_WARN_PCT",
    "CLAUDE_BUDGET_LOOP_LIMIT",
    "CLAUDE_BUDGET_CONFIG",
    "CLAUDE_BUDGET_OUTPUT",
]


@contextmanager
def env(**overrides):
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in overrides.items():
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


def invoke(payload, **env_overrides):
    with env(**env_overrides):
        stdin = io.StringIO(json.dumps(payload) if payload is not None else "")
        stdout, stderr = io.StringIO(), io.StringIO()
        code = run(stdin, stdout, stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def payload_for(path, tool_name="Bash", tool_input=None):
    return {
        "session_id": "s1",
        "transcript_path": path,
        "cwd": "/tmp",
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": "echo hi"} if tool_input is None else tool_input,
    }


# The exact call that repeats inside loop.jsonl / loop3.jsonl. Under correct
# PreToolUse semantics the call being evaluated is folded into the trailing run,
# so blocking a loop requires the current call to match the repeated signature.
PYTEST_CALL = {"command": "pytest -q"}


class TestTokenBudget(unittest.TestCase):
    def test_under_budget_allows(self):
        code, _out, _err = invoke(payload_for(DEDUP), CLAUDE_BUDGET_MAX_TOKENS=100000)
        self.assertEqual(code, EXIT_ALLOW)

    def test_over_budget_blocks_exit2_with_stderr(self):
        code, out, err = invoke(payload_for(DEDUP), CLAUDE_BUDGET_MAX_TOKENS=5000)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertEqual(out, "")
        self.assertIn("Token budget exceeded", err)
        self.assertIn("budget-guard", err)

    def test_warn_allows_but_writes_stderr(self):
        # dedup total is 7300; 80% of 8000 = 6400 -> warn, still allow.
        code, out, err = invoke(payload_for(DEDUP), CLAUDE_BUDGET_MAX_TOKENS=8000)
        self.assertEqual(code, EXIT_ALLOW)
        self.assertIn("WARNING", err)


class TestUsdBudget(unittest.TestCase):
    def test_usd_over_blocks(self):
        # dedup opus cost ~= $0.0619
        code, _out, err = invoke(payload_for(DEDUP), CLAUDE_BUDGET_MAX_USD=0.05)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertIn("USD budget exceeded", err)

    def test_usd_under_allows(self):
        code, _out, _err = invoke(payload_for(DEDUP), CLAUDE_BUDGET_MAX_USD=1.0)
        self.assertEqual(code, EXIT_ALLOW)


class TestLoopDetection(unittest.TestCase):
    def test_loop_blocks(self):
        # loop.jsonl has 4 identical Bash 'pytest -q' calls; the current call
        # matches, so the trailing run is 5 >= limit 4 -> block.
        code, _out, err = invoke(
            payload_for(LOOP, tool_input=PYTEST_CALL), CLAUDE_BUDGET_LOOP_LIMIT=4
        )
        self.assertEqual(code, EXIT_BLOCK)
        self.assertIn("Retry-loop detected", err)

    def test_loop_below_limit_allows(self):
        code, _out, _err = invoke(
            payload_for(LOOP, tool_input=PYTEST_CALL), CLAUDE_BUDGET_LOOP_LIMIT=10
        )
        self.assertEqual(code, EXIT_ALLOW)


class TestCurrentCallLoopParticipation(unittest.TestCase):
    """FIX 1: the PreToolUse call under evaluation participates in loop detection."""

    def test_current_identical_call_completes_the_loop(self):
        # loop3.jsonl holds only 3 identical calls (< limit 4): the transcript
        # ALONE would NOT block. Folding in the current identical call makes the
        # run 4 == limit -> block, pre-execution. This proves the current call
        # participates.
        code, _out, err = invoke(
            payload_for(LOOP3, tool_input=PYTEST_CALL), CLAUDE_BUDGET_LOOP_LIMIT=4
        )
        self.assertEqual(code, EXIT_BLOCK)
        self.assertIn("Retry-loop detected", err)
        # Count reflects the 3 transcript calls + the current one.
        self.assertIn("4 times", err)

    def test_different_call_after_repeats_is_not_false_blocked(self):
        # loop.jsonl holds 4 identical calls == limit 4. The OLD (buggy) code
        # would block a *different* current tool because the transcript tail
        # already held 4 repeats. The current call breaks the run -> allow.
        code, _out, _err = invoke(
            payload_for(LOOP, tool_name="Read", tool_input={"file": "x"}),
            CLAUDE_BUDGET_LOOP_LIMIT=4,
        )
        self.assertEqual(code, EXIT_ALLOW)

    def test_malformed_current_call_does_not_crash(self):
        # A non-dict tool_input / non-str tool_name must be skipped, never raise;
        # loop detection then falls back to the transcript-only trailing run.
        payload = payload_for(LOOP3)
        payload["tool_name"] = 123          # invalid
        payload["tool_input"] = "not-a-dict"  # invalid
        code, _out, _err = invoke(payload, CLAUDE_BUDGET_LOOP_LIMIT=4)
        # Transcript alone has 3 < 4 and the bad current call is skipped -> allow.
        self.assertEqual(code, EXIT_ALLOW)

    def test_invalid_current_call_does_not_block_on_stale_tail(self):
        # FIX 1: loop.jsonl already holds 4 identical calls == limit 4. With an
        # INVALID current call (tool_input not a dict) nothing valid is appended,
        # so there is NO proven identical continuation. The OLD code still ran
        # detect_loop against the stale tail (4 >= 4) and BLOCKED a call never
        # shown to match — an absolute fail-open violation. Now: unappended
        # current call -> loop UNPROVEN -> allow (warn at most), never block.
        payload = payload_for(LOOP)  # 4 identical calls at the tail
        payload["tool_name"] = "Bash"
        payload["tool_input"] = "not-a-dict"  # invalid -> nothing appended
        code, _out, err = invoke(payload, CLAUDE_BUDGET_LOOP_LIMIT=4)
        self.assertEqual(code, EXIT_ALLOW)
        self.assertNotIn("Retry-loop detected", err)

    def test_missing_tool_name_does_not_block_on_stale_tail(self):
        # FIX 1 sibling: a MISSING tool_name (with a valid tool_input) is also an
        # invalid current call -> not appended -> loop cannot block off the stale
        # 4-repeat tail.
        payload = payload_for(LOOP, tool_input=PYTEST_CALL)
        del payload["tool_name"]
        code, _out, err = invoke(payload, CLAUDE_BUDGET_LOOP_LIMIT=4)
        self.assertEqual(code, EXIT_ALLOW)
        self.assertNotIn("Retry-loop detected", err)


class TestJsonOutputMode(unittest.TestCase):
    def test_block_as_json_stdout_exit0(self):
        # FIX C: json mode emits the OFFICIAL PreToolUse contract
        # (hookSpecificOutput.permissionDecision == "deny"), NOT the deprecated
        # top-level {"decision":"block"} that PreToolUse may ignore.
        code, out, err = invoke(
            payload_for(DEDUP),
            CLAUDE_BUDGET_MAX_TOKENS=5000,
            CLAUDE_BUDGET_OUTPUT="json",
        )
        self.assertEqual(code, EXIT_ALLOW)  # json mode never exits 2
        self.assertEqual(err, "")
        data = json.loads(out)
        # Deprecated form must be gone.
        self.assertNotIn("decision", data)
        hso = data["hookSpecificOutput"]
        self.assertEqual(hso["hookEventName"], "PreToolUse")
        self.assertEqual(hso["permissionDecision"], "deny")
        self.assertIn("Token budget exceeded", hso["permissionDecisionReason"])

    def test_allow_as_json_emits_nothing_exit0(self):
        # A non-blocked call in json mode emits no deny payload and exits 0.
        code, out, err = invoke(
            payload_for(DEDUP),
            CLAUDE_BUDGET_MAX_TOKENS=100000,
            CLAUDE_BUDGET_OUTPUT="json",
        )
        self.assertEqual(code, EXIT_ALLOW)
        self.assertEqual(out, "")


def _write_transcript(lines):
    """Write JSONL lines to a temp file; return its path (caller unlinks)."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as fh:
        for line in lines:
            fh.write(line)
            if not line.endswith("\n"):
                fh.write("\n")
        return fh.name


def _usage_line(request_id, **usage):
    return json.dumps(
        {
            "type": "assistant",
            "requestId": request_id,
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "usage": usage,
            },
        }
    )


class TestHugeIntegerOverflow(unittest.TestCase):
    """FIX A: an enormous finite int must not OverflowError into a fail-open."""

    def test_huge_int_blocks_with_token_and_usd_limits(self):
        # A single line whose input_tokens is astronomically large. It used to
        # crash pricing (int*float OverflowError) -> fail-open ALLOW, bypassing
        # BOTH ceilings. Now: token clamp + inf-on-overflow -> BLOCK.
        path = _write_transcript([_usage_line("req_huge", input_tokens=10 ** 400,
                                              output_tokens=5)])
        try:
            code, _out, err = invoke(
                payload_for(path),
                CLAUDE_BUDGET_MAX_TOKENS=1_000_000,
                CLAUDE_BUDGET_MAX_USD=10,
            )
            self.assertEqual(code, EXIT_BLOCK)
            # A breach is confirmed (either ceiling suffices).
            self.assertTrue(
                "Token budget exceeded" in err or "USD budget exceeded" in err
            )
        finally:
            os.unlink(path)

    def test_huge_int_blocks_with_only_token_limit(self):
        path = _write_transcript([_usage_line("req_huge", input_tokens=10 ** 400)])
        try:
            code, _out, err = invoke(
                payload_for(path), CLAUDE_BUDGET_MAX_TOKENS=1_000_000
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
        finally:
            os.unlink(path)

    def test_huge_int_blocks_with_only_usd_limit(self):
        # Only a USD ceiling: the clamped-huge token count prices well over any
        # finite USD limit -> block on USD (never a swallowed overflow-allow).
        path = _write_transcript([_usage_line("req_huge", input_tokens=10 ** 400)])
        try:
            code, _out, err = invoke(payload_for(path), CLAUDE_BUDGET_MAX_USD=10)
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("USD budget exceeded", err)
        finally:
            os.unlink(path)


# A physical line longer than the DEFAULT 4 MiB per-line cap: dropped by the
# bounded reader, which flags the parse ``partial``.
_OVERLONG_LINE = "x" * (5 * 1024 * 1024)
_PARTIAL_WARNING = "transcript too large to fully account"


class TestPartialParse(unittest.TestCase):
    """FIX B: an incomplete parse must never silently pass as a full total."""

    def test_breach_before_bound_still_blocks(self):
        # Over-budget spend sits BEFORE the overlong (dropped) line. The single
        # pass counts it, confirms the breach, and blocks despite partial data.
        path = _write_transcript([
            _usage_line("req_big", input_tokens=10_000, output_tokens=0),
            _OVERLONG_LINE,
        ])
        try:
            code, _out, err = invoke(
                payload_for(path), CLAUDE_BUDGET_MAX_TOKENS=5000
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
        finally:
            os.unlink(path)

    def test_partial_no_breach_warns_and_allows(self):
        # The read portion is UNDER budget but a bound tripped: we cannot prove a
        # breach (fail-open forbids blocking) yet must not hide the incompleteness
        # -> warn on stderr, exit 0.
        path = _write_transcript([
            _usage_line("req_small", input_tokens=100, output_tokens=10),
            _OVERLONG_LINE,
        ])
        try:
            code, out, err = invoke(
                payload_for(path), CLAUDE_BUDGET_MAX_TOKENS=1_000_000
            )
            self.assertEqual(code, EXIT_ALLOW)
            self.assertEqual(out, "")
            self.assertIn(_PARTIAL_WARNING, err)
        finally:
            os.unlink(path)

    def test_partial_plus_breach_blocks_not_warns(self):
        # Partial AND over budget on the read portion -> a confirmed breach still
        # blocks; the partial-warning allow path must NOT swallow it.
        path = _write_transcript([
            _usage_line("req_big", input_tokens=50_000, output_tokens=0),
            _OVERLONG_LINE,
        ])
        try:
            code, _out, err = invoke(
                payload_for(path), CLAUDE_BUDGET_MAX_TOKENS=5000
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
            self.assertNotIn(_PARTIAL_WARNING, err)
        finally:
            os.unlink(path)


def _tool_line(request_id, cmd="pytest -q"):
    """Assistant line with ONE Bash tool_use block (for loop-run fixtures)."""
    return json.dumps({
        "type": "assistant",
        "requestId": request_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "name": "Bash",
                         "input": {"command": cmd}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    })


class TestLoopTrustworthiness(unittest.TestCase):
    """FIX 1 + FIX 2: a loop blocks ONLY when the trailing-run TAIL suffix is fully
    known — a mid-stream gap breaks the run but does NOT poison trust, and a
    usage/dedup bound (``partial``) does NOT disable the loop; only a genuinely
    unknown suffix (truncation / read error / missing) leaves the loop unproven."""

    def test_loop_after_overlong_gap_still_blocks_despite_partial(self):
        # FIX 1 regression: an overlong (dropped) line precedes a trailing run of
        # identical calls fully observed to EOF. The overlong line makes the read
        # PARTIAL (usage incomplete) but its drain RESYNCS to the next newline, so
        # the tail suffix is KNOWN. The fresh run after the gap (4) + the current
        # identical call (=5) reaches the limit on a trustworthy tail -> BLOCK.
        # The OLD code stickily poisoned the loop after any dropped line and wrongly
        # ALLOWED here; ``partial`` alone must NOT disable a proven loop.
        path = _write_transcript([
            _OVERLONG_LINE,
            _tool_line("r1"), _tool_line("r2"),
            _tool_line("r3"), _tool_line("r4"),
        ])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=4,
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_loop_after_malformed_gap_reaching_eof_blocks(self):
        # FIX 1 KEY regression: a malformed line, then N identical calls fully
        # observed to EOF. The gap breaks the earlier run but does NOT poison the
        # tail, so the clean post-gap run (3 transcript calls) + current identical
        # call (=4) reaches the limit and BLOCKS. Under the OLD sticky-untrustworthy
        # model this was permanently disabled and wrongly allowed.
        path = _write_transcript([
            "this is { not json",
            _tool_line("r1"), _tool_line("r2"), _tool_line("r3"),
        ])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=4,
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Retry-loop detected", err)
            self.assertIn("4 times", err)
        finally:
            os.unlink(path)

    def test_malformed_gap_breaks_run_no_false_block(self):
        # Two identical calls separated by a malformed line. Without the gap-reset
        # the transcript tail (2) + the current identical call (=3) would hit the
        # limit and block. The gap breaks the run -> only 1 real trailing call +
        # current = 2 < 3 -> allow. No false "Retry-loop detected".
        path = _write_transcript([
            _tool_line("r1"),
            "this is { not json",
            _tool_line("r2"),
        ])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=3,
            )
            self.assertEqual(code, EXIT_ALLOW)
            self.assertNotIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_consecutive_run_no_skip_still_blocks(self):
        # Control: the same shape WITHOUT the malformed gap -> 2 trailing calls +
        # current identical call = 3 == limit on a complete, trustworthy read ->
        # block. Proves the reset (not some unrelated change) is what allows above.
        path = _write_transcript([_tool_line("r1"), _tool_line("r2")])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=3,
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_bad_tool_use_input_gap_breaks_run_no_false_block(self):
        # FIX 2 (record level): a tool_use block with a NON-DICT input between two
        # identical calls. Without the record-level reset the tail (2) + current
        # identical call (=3) would hit limit 3 and block. The un-interpretable
        # tool_use breaks the run -> allow, no false loop.
        bad_input = json.dumps({
            "type": "assistant", "requestId": "rx",
            "message": {"role": "assistant", "model": "claude-opus-4-8",
                        "content": [{"type": "tool_use", "name": "Bash",
                                     "input": "NOT-A-DICT"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1}},
        })
        path = _write_transcript([_tool_line("r1"), bad_input, _tool_line("r2")])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=3,
            )
            self.assertEqual(code, EXIT_ALLOW)
            self.assertNotIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_non_list_content_gap_breaks_run_no_false_block(self):
        # FIX 2 (record level): an assistant line whose content is not a list
        # between two identical calls breaks the run -> allow, no false loop.
        bad_content = json.dumps({
            "type": "assistant", "requestId": "ry",
            "message": {"role": "assistant", "model": "claude-opus-4-8",
                        "content": "string not a list",
                        "usage": {"input_tokens": 10, "output_tokens": 1}},
        })
        path = _write_transcript([_tool_line("r1"), bad_content, _tool_line("r2")])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=3,
            )
            self.assertEqual(code, EXIT_ALLOW)
            self.assertNotIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_plain_text_interleave_still_blocks(self):
        # Control: pure-text assistant lines interleaved between identical calls
        # do NOT break the run (normal per-block interleaving), so 2 trailing + 1
        # current identical call = 3 == limit on a trustworthy read -> block.
        text_line = json.dumps({
            "type": "assistant", "requestId": "t1",
            "message": {"role": "assistant", "model": "claude-opus-4-8",
                        "content": [{"type": "text", "text": "hmm"}],
                        "usage": {"input_tokens": 10, "output_tokens": 1}},
        })
        path = _write_transcript([_tool_line("r1"), text_line, _tool_line("r2")])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=3,
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Retry-loop detected", err)
        finally:
            os.unlink(path)

    def test_token_breach_on_partial_read_still_blocks_not_loop(self):
        # A partial+untrustworthy read that ALSO crosses the token ceiling on the
        # read portion still blocks — on TOKENS, never on the unproven loop.
        path = _write_transcript([
            _tool_line("r1"), _tool_line("r2"),
            _tool_line("r3"), _tool_line("r4"),
            _OVERLONG_LINE,
        ])
        try:
            code, _out, err = invoke(
                payload_for(path, tool_input=PYTEST_CALL),
                CLAUDE_BUDGET_LOOP_LIMIT=4,
                CLAUDE_BUDGET_MAX_TOKENS=200,
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
            self.assertNotIn("Retry-loop detected", err)
        finally:
            os.unlink(path)


class TestReadFailureLoopUntrustworthy(unittest.TestCase):
    """FIX 1: a transcript READ ERROR must not count as a trustworthy empty read.

    An unavailable transcript (missing path, non-regular file) leaves the trailing
    run's SUFFIX unknown, not a proven-empty run. With the current call folded in
    the count would reach 1, so ``CLAUDE_BUDGET_LOOP_LIMIT=1`` used to BLOCK the
    very first call — an absolute fail-open violation. The read failure now marks
    the run ``tail_suffix_unknown``, forbidding a loop BLOCK (warn/allow at most).
    """

    def test_missing_transcript_loop_limit_1_allows(self):
        code, _out, err = invoke(
            payload_for("/nonexistent/path/session.jsonl", tool_input=PYTEST_CALL),
            CLAUDE_BUDGET_LOOP_LIMIT=1,
        )
        self.assertEqual(code, EXIT_ALLOW)
        self.assertNotIn("Retry-loop detected", err)

    def test_missing_transcript_path_key_loop_limit_1_allows(self):
        # No transcript_path key at all (invalid path) + loop_limit 1 -> allow.
        code, _out, err = invoke(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": PYTEST_CALL,
            },
            CLAUDE_BUDGET_LOOP_LIMIT=1,
        )
        self.assertEqual(code, EXIT_ALLOW)
        self.assertNotIn("Retry-loop detected", err)

    def test_fifo_transcript_loop_limit_1_allows(self):
        # A non-regular (FIFO) transcript path + loop_limit 1 must ALLOW without
        # hanging. Guarded by a daemon thread + join timeout so a regression that
        # blocks on the FIFO read cannot wedge the suite.
        import tempfile
        import threading

        d = tempfile.mkdtemp()
        fifo = os.path.join(d, "pipe")
        os.mkfifo(fifo)

        result = {}

        def worker():
            try:
                code, _out, err = invoke(
                    payload_for(fifo, tool_input=PYTEST_CALL),
                    CLAUDE_BUDGET_LOOP_LIMIT=1,
                )
                result["code"] = code
                result["err"] = err
            except Exception as exc:  # pragma: no cover - defensive
                result["exc"] = exc

        try:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "hook hung on a FIFO transcript")
            self.assertNotIn("exc", result)
            self.assertEqual(result.get("code"), EXIT_ALLOW)
            self.assertNotIn("Retry-loop detected", result.get("err", ""))
        finally:
            os.unlink(fifo)
            os.rmdir(d)


def _decide_with_caps(path, config, current_input, **caps):
    """Mirror ``hook.evaluate`` exactly, but stream with explicit bounds ``caps``.

    ``stream_session`` caps (``max_lines`` / ``max_total_bytes`` / dedup / model)
    are not plumbed through the hook's env config, so this helper reproduces the
    real evaluate pipeline — stream -> append current call -> detect_loop_run ->
    ``loop_trustworthy = (not tail_suffix_unknown) and appended`` -> decide — to
    exercise truncation / usage-bound cases end to end. Returns a ``Decision``.
    """
    from budget_guard.transcript import stream_session, ToolCall
    from budget_guard.loops import detect_loop_run
    from budget_guard.budget import decide

    session, run, _partial = stream_session(path, **caps)
    tokens = session.total_tokens
    appended = False
    if config.loop_limit and isinstance(current_input, dict):
        run.add(ToolCall(name="Bash", input=current_input))
        appended = True
    loop = detect_loop_run(run, config.loop_limit)
    loop_trustworthy = (not run.tail_suffix_unknown) and appended
    return decide(tokens=tokens, usd=0.0, config=config,
                  loop=loop, loop_trustworthy=loop_trustworthy)


class TestLoopWithStreamCaps(unittest.TestCase):
    """FIX 1 end-to-end (evaluate pipeline) for cases needing explicit caps:
    truncation disables the loop but not a token breach; a usage bound does NOT
    disable a fully-observed loop."""

    def test_truncated_read_loop_not_blocked_but_token_breach_blocks(self):
        # Required test #2: a trailing identical run truncated by ``max_lines`` has
        # an UNKNOWN suffix -> the loop is unproven (warn, not block). But the
        # token total on the read portion is over ceiling -> BLOCK on tokens, never
        # on the loop (tokens are monotonic).
        from budget_guard.config import Config
        # Each _tool_line = 120 tokens; 4 read lines = 480 tokens.
        path = _write_transcript([_tool_line("r%d" % i) for i in range(12)])
        try:
            cfg = Config(max_tokens=200, loop_limit=4)
            d = _decide_with_caps(path, cfg, PYTEST_CALL, max_lines=4)
            self.assertEqual(d.action, "block")
            self.assertIn("Token budget exceeded", d.message)
            self.assertNotIn("Retry-loop detected", d.message)
        finally:
            os.unlink(path)

    def test_truncated_read_loop_alone_does_not_block(self):
        # Required test #2 (loop half): same truncation, NO token ceiling -> the
        # unproven loop may only WARN, never block.
        from budget_guard.config import Config
        path = _write_transcript([_tool_line("r%d" % i) for i in range(12)])
        try:
            cfg = Config(loop_limit=4)
            d = _decide_with_caps(path, cfg, PYTEST_CALL, max_lines=4)
            self.assertNotEqual(d.action, "block")
            self.assertNotIn("Retry-loop detected", d.message)
        finally:
            os.unlink(path)

    def test_dedup_cap_usage_partial_loop_still_blocks(self):
        # Required test #3: a dedup-key cap makes the read usage-partial, but every
        # tool_use block is still folded to EOF, so the tail is KNOWN. The loop is
        # fully observed and BLOCKS despite the usage bound.
        from budget_guard.config import Config
        # 3 transcript calls + current identical = 4 == limit.
        path = _write_transcript([_tool_line("r1"), _tool_line("r2"), _tool_line("r3")])
        try:
            cfg = Config(loop_limit=4)
            d = _decide_with_caps(path, cfg, PYTEST_CALL, max_dedup_keys=1)
            self.assertEqual(d.action, "block")
            self.assertIn("Retry-loop detected", d.message)
        finally:
            os.unlink(path)

    def test_model_cap_usage_partial_loop_still_blocks(self):
        # Required test #3 sibling: a distinct-model cap (usage-partial) likewise
        # must NOT disable a fully-observed loop. The three identical Bash calls
        # carry DISTINCT models so ``max_models=1`` genuinely trips (later models
        # fold to the sentinel), yet the trailing run is intact to EOF.
        from budget_guard.config import Config

        def line(rid, model):
            return json.dumps({
                "type": "assistant", "requestId": rid,
                "message": {"role": "assistant", "model": model,
                            "content": [{"type": "tool_use", "name": "Bash",
                                         "input": {"command": "pytest -q"}}],
                            "usage": {"input_tokens": 100, "output_tokens": 20}},
            })
        path = _write_transcript([line("r1", "m-a"), line("r2", "m-b"),
                                  line("r3", "m-c")])
        try:
            cfg = Config(loop_limit=4)
            d = _decide_with_caps(path, cfg, PYTEST_CALL, max_models=1)
            self.assertEqual(d.action, "block")
            self.assertIn("Retry-loop detected", d.message)
        finally:
            os.unlink(path)


class _RaisingStream:
    """A write-only stream whose ``write`` always raises ``OSError`` (broken pipe)."""

    def __init__(self):
        self.attempts = 0

    def write(self, *_args, **_kwargs):
        self.attempts += 1
        raise OSError("broken stream")

    def flush(self):  # pragma: no cover - defensive
        raise OSError("broken stream")


class _FlushRaisingStream:
    """A stream whose ``write`` SUCCEEDS (buffers) but ``flush`` raises ``OSError``.

    Models a buffered ``sys.stdout``: ``write`` only fills the user-space buffer
    and reports success, then the (deferred) ``flush`` fails before the bytes ever
    reach the pipe. FIX 2 must treat this as a delivery failure.
    """

    def __init__(self):
        self.written = ""
        self.flushes = 0

    def write(self, text):
        # Accept the whole payload (report full length -> not a short write) so the
        # ONLY failure signal is the flush below.
        self.written += text
        return len(text)

    def flush(self):
        self.flushes += 1
        raise OSError("flush failed after buffered write")


class _ShortWriteStream:
    """A stream whose ``write`` accepts only PART of the payload (short write)."""

    def __init__(self):
        self.written = ""
        self.flushes = 0

    def write(self, text):
        # Only the first character lands; the tail never does -> delivery failure.
        head = text[:1]
        self.written += head
        return len(head)

    def flush(self):  # pragma: no cover - not reached (short write fails first)
        self.flushes += 1


class TestOutputErrorDoesNotDowngradeBlock(unittest.TestCase):
    """FIX 2: an OUTPUT error must not turn a confirmed block into an allow."""

    def test_stderr_oserror_still_exits_block(self):
        # Exit mode: DEDUP over a 5000-token ceiling -> confirmed block. A stderr
        # that raises OSError must not swallow it into fail-open; exit 2 is the
        # floor regardless of whether the reason was written.
        with env(CLAUDE_BUDGET_MAX_TOKENS="5000"):
            stdin = io.StringIO(json.dumps(payload_for(DEDUP)))
            stderr = _RaisingStream()
            code = run(stdin, io.StringIO(), stderr)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertGreaterEqual(stderr.attempts, 1)  # a best-effort write happened

    def test_json_stdout_oserror_falls_back_to_block(self):
        # JSON mode: writing the deny payload to a broken stdout must NOT exit 0
        # (which would let the blocked call through). Fall back to exit 2 with a
        # best-effort stderr reason.
        with env(CLAUDE_BUDGET_MAX_TOKENS="5000", CLAUDE_BUDGET_OUTPUT="json"):
            stdin = io.StringIO(json.dumps(payload_for(DEDUP)))
            stderr = io.StringIO()
            code = run(stdin, _RaisingStream(), stderr)
        self.assertEqual(code, EXIT_BLOCK)
        # Fallback reason reached the (working) stderr.
        self.assertIn("Token budget exceeded", stderr.getvalue())

    def test_json_stdout_flush_oserror_falls_back_to_block(self):
        # FIX 2 core: JSON mode where write() SUCCEEDS (buffers) but flush() raises
        # OSError. A successful write is NOT proof of delivery — the deny packet
        # never reached the pipe — so a confirmed block must NOT exit 0. It falls
        # back to exit 2 with a best-effort stderr reason.
        stdout = _FlushRaisingStream()
        with env(CLAUDE_BUDGET_MAX_TOKENS="5000", CLAUDE_BUDGET_OUTPUT="json"):
            stdin = io.StringIO(json.dumps(payload_for(DEDUP)))
            stderr = io.StringIO()
            code = run(stdin, stdout, stderr)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertGreaterEqual(stdout.flushes, 1)   # a synchronous flush was tried
        self.assertIn("Token budget exceeded", stderr.getvalue())

    def test_json_stdout_short_write_falls_back_to_block(self):
        # FIX 2: a SHORT write (only part of the deny payload accepted) is a
        # delivery failure too -> fall back to exit 2, best-effort stderr reason.
        with env(CLAUDE_BUDGET_MAX_TOKENS="5000", CLAUDE_BUDGET_OUTPUT="json"):
            stdin = io.StringIO(json.dumps(payload_for(DEDUP)))
            stderr = io.StringIO()
            code = run(stdin, _ShortWriteStream(), stderr)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertIn("Token budget exceeded", stderr.getvalue())

    def test_warn_stderr_oserror_stays_allow(self):
        # A WARNING is fail-open: a stderr that raises must NOT be hardened to a
        # block — it stays exit 0. (dedup total 7300; 80% of 8000 = 6400 -> warn.)
        with env(CLAUDE_BUDGET_MAX_TOKENS="8000"):
            stdin = io.StringIO(json.dumps(payload_for(DEDUP)))
            code = run(stdin, io.StringIO(), _RaisingStream())
        self.assertEqual(code, EXIT_ALLOW)


class TestHugeFiniteTokenCeilingNotUnderCounted(unittest.TestCase):
    """FIX 3: a huge FINITE token count must trip the token ceiling, not be
    clamped down under it."""

    def test_huge_finite_tokens_block_on_token_ceiling(self):
        # input_tokens = 10**15 with a token ceiling of 10**14 (both above the old
        # 10**12 clamp). The old clamp lowered 10**15 to 10**12 < 10**14 and
        # ALLOWED — a fail-under-count. Now the exact value blocks on TOKENS.
        path = _write_transcript([_usage_line("req_big", input_tokens=10 ** 15,
                                              output_tokens=0)])
        try:
            code, _out, err = invoke(
                payload_for(path), CLAUDE_BUDGET_MAX_TOKENS=str(10 ** 14)
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
        finally:
            os.unlink(path)


class TestFailOpen(unittest.TestCase):
    def test_no_limits_configured_is_noop(self):
        # Huge usage, but no limits set -> allow. Never surprise-block.
        code, out, err = invoke(payload_for(DEDUP))
        self.assertEqual(code, EXIT_ALLOW)
        self.assertEqual((out, err), ("", ""))

    def test_missing_transcript_allows(self):
        code, _out, _err = invoke(
            payload_for("/nonexistent/path/session.jsonl"),
            CLAUDE_BUDGET_MAX_TOKENS=1,
        )
        self.assertEqual(code, EXIT_ALLOW)

    def test_empty_transcript_allows(self):
        code, _out, _err = invoke(payload_for(EMPTY), CLAUDE_BUDGET_MAX_TOKENS=1)
        self.assertEqual(code, EXIT_ALLOW)

    def test_empty_stdin_allows(self):
        code, _out, _err = invoke(None, CLAUDE_BUDGET_MAX_TOKENS=1)
        self.assertEqual(code, EXIT_ALLOW)

    def test_garbage_stdin_allows(self):
        with env(CLAUDE_BUDGET_MAX_TOKENS="1"):
            stdin = io.StringIO("}{ not json")
            stdout, stderr = io.StringIO(), io.StringIO()
            code = run(stdin, stdout, stderr)
        self.assertEqual(code, EXIT_ALLOW)

    def test_missing_transcript_path_key_allows(self):
        code, _out, _err = invoke(
            {"hook_event_name": "PreToolUse"}, CLAUDE_BUDGET_MAX_TOKENS=1
        )
        self.assertEqual(code, EXIT_ALLOW)


class TestConfigFile(unittest.TestCase):
    def test_json_config_file_limits(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump({"max_tokens": 5000}, fh)
            cfg_path = fh.name
        try:
            code, _out, err = invoke(
                payload_for(DEDUP), CLAUDE_BUDGET_CONFIG=cfg_path
            )
            self.assertEqual(code, EXIT_BLOCK)
            self.assertIn("Token budget exceeded", err)
        finally:
            os.unlink(cfg_path)

    def test_env_overrides_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump({"max_tokens": 5000}, fh)
            cfg_path = fh.name
        try:
            # File says 5000 (would block), env raises it to 100000 (allow).
            code, _out, _err = invoke(
                payload_for(DEDUP),
                CLAUDE_BUDGET_CONFIG=cfg_path,
                CLAUDE_BUDGET_MAX_TOKENS=100000,
            )
            self.assertEqual(code, EXIT_ALLOW)
        finally:
            os.unlink(cfg_path)


class TestOversizedStdinFailsOpen(unittest.TestCase):
    def test_oversized_stdin_payload_allows(self):
        import io
        from budget_guard.hook import run, MAX_STDIN_BYTES, EXIT_ALLOW
        # A payload larger than the cap must fail-open (allow), not OOM/block.
        huge = '{"tool_name":"Bash","tool_input":{"x":"' + ("a" * (MAX_STDIN_BYTES + 10)) + '"}}'
        rc = run(io.StringIO(huge), io.StringIO(), io.StringIO())
        self.assertEqual(rc, EXIT_ALLOW)


if __name__ == "__main__":
    unittest.main()
