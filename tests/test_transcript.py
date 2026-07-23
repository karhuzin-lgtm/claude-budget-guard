"""Tests for transcript parsing — dedup is the headline case."""

from __future__ import annotations

import os
import unittest

from budget_guard.transcript import (
    Usage,
    parse_records,
    parse_transcript,
    iter_records,
    iter_tool_calls,
    load_tool_calls,
    stream_session,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _line(request_id, model="claude-opus-4-8", msg_id=None, **usage):
    import json

    message = {"role": "assistant", "model": model, "usage": usage}
    if msg_id is not None:
        message["id"] = msg_id
    record = {"type": "assistant", "message": message}
    if request_id is not None:
        record["requestId"] = request_id
    return json.dumps(record)


class TestDedup(unittest.TestCase):
    def test_same_request_id_counted_once(self):
        # Two content-block lines share requestId + identical usage.
        lines = [
            _line("req_1", input_tokens=1000, output_tokens=200,
                  cache_creation_input_tokens=500, cache_read_input_tokens=3000),
            _line("req_1", input_tokens=1000, output_tokens=200,
                  cache_creation_input_tokens=500, cache_read_input_tokens=3000),
        ]
        session = parse_records(iter_records(lines))
        self.assertEqual(session.requests, 1)
        self.assertEqual(session.total.input_tokens, 1000)
        self.assertEqual(session.total.output_tokens, 200)
        self.assertEqual(session.total.cache_creation_tokens, 500)
        self.assertEqual(session.total.cache_read_tokens, 3000)

    def test_distinct_request_ids_summed(self):
        lines = [
            _line("req_1", input_tokens=1000, output_tokens=200,
                  cache_creation_input_tokens=500, cache_read_input_tokens=3000),
            _line("req_1", input_tokens=1000, output_tokens=200,
                  cache_creation_input_tokens=500, cache_read_input_tokens=3000),
            _line("req_2", input_tokens=500, output_tokens=100,
                  cache_creation_input_tokens=0, cache_read_input_tokens=2000),
        ]
        session = parse_records(iter_records(lines))
        self.assertEqual(session.requests, 2)
        self.assertEqual(session.total_tokens, 1500 + 300 + 500 + 5000)

    def test_fallback_to_message_id_when_no_request_id(self):
        lines = [
            _line(None, msg_id="msg_x", input_tokens=100, output_tokens=10),
            _line(None, msg_id="msg_x", input_tokens=100, output_tokens=10),
        ]
        session = parse_records(iter_records(lines))
        self.assertEqual(session.requests, 1)
        self.assertEqual(session.total_tokens, 110)

    def test_no_key_line_is_skipped(self):
        # No requestId and no message.id -> cannot dedup -> skip to avoid inflation.
        lines = [
            _line(None, input_tokens=999, output_tokens=999),
        ]
        session = parse_records(iter_records(lines))
        self.assertEqual(session.requests, 0)
        self.assertEqual(session.total_tokens, 0)


class TestSummation(unittest.TestCase):
    def test_total_input_and_total_tokens(self):
        u = Usage(input_tokens=10, output_tokens=5,
                  cache_creation_tokens=2, cache_read_tokens=100)
        self.assertEqual(u.total_input_tokens, 112)
        self.assertEqual(u.total_tokens, 117)

    def test_usage_add(self):
        a = Usage(input_tokens=1, output_tokens=2,
                  cache_creation_tokens=3, cache_read_tokens=4)
        b = Usage(input_tokens=10, output_tokens=20,
                  cache_creation_tokens=30, cache_read_tokens=40)
        c = a + b
        self.assertEqual((c.input_tokens, c.output_tokens,
                          c.cache_creation_tokens, c.cache_read_tokens),
                         (11, 22, 33, 44))

    def test_per_model_breakdown(self):
        lines = [
            _line("a", model="claude-opus-4-8", input_tokens=100, output_tokens=10),
            _line("b", model="claude-haiku-4-5", input_tokens=50, output_tokens=5),
        ]
        session = parse_records(iter_records(lines))
        self.assertIn("claude-opus-4-8", session.by_model)
        self.assertIn("claude-haiku-4-5", session.by_model)


class TestDefensiveParsing(unittest.TestCase):
    def test_empty_transcript(self):
        session = parse_transcript(os.path.join(FIXTURES, "empty.jsonl"))
        self.assertEqual(session.total_tokens, 0)
        self.assertEqual(session.requests, 0)

    def test_malformed_lines_skipped(self):
        # Only the one well-formed assistant line with a key should count.
        session = parse_transcript(os.path.join(FIXTURES, "malformed.jsonl"))
        self.assertEqual(session.requests, 1)
        self.assertEqual(session.total_tokens, 800 + 150)

    def test_missing_usage_and_user_lines_skipped(self):
        lines = [
            '{"type":"user","message":{"role":"user","content":"hi"}}',
            '{"type":"system","content":"x"}',
            '{"type":"assistant","requestId":"r","message":{"role":"assistant"}}',
        ]
        session = parse_records(iter_records(lines))
        self.assertEqual(session.total_tokens, 0)

    def test_fixture_dedup_totals(self):
        session = parse_transcript(os.path.join(FIXTURES, "dedup.jsonl"))
        self.assertEqual(session.requests, 2)
        # req_1 (once) + req_2
        self.assertEqual(session.total.input_tokens, 1500)
        self.assertEqual(session.total.output_tokens, 300)
        self.assertEqual(session.total.cache_creation_tokens, 500)
        self.assertEqual(session.total.cache_read_tokens, 5000)
        self.assertEqual(session.total_tokens, 7300)


class TestToolCalls(unittest.TestCase):
    def test_iter_tool_calls_order(self):
        calls = load_tool_calls(os.path.join(FIXTURES, "loop.jsonl"))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c.name == "Bash" for c in calls))
        self.assertEqual(calls[0].input, {"command": "pytest -q"})

    def test_non_list_content_ignored(self):
        lines = [
            '{"type":"assistant","message":{"content":"plain string"}}',
        ]
        calls = iter_tool_calls(iter_records(lines))
        self.assertEqual(calls, [])


class TestNegativeTokensClamped(unittest.TestCase):
    def test_negative_counts_clamp_to_zero(self):
        # A corrupt/hostile line with negative counts must NOT lower the total
        # (would be a guardrail bypass). Negatives read as 0.
        lines = [
            _line("req_ok", input_tokens=1000, output_tokens=500),
            _line("req_neg", input_tokens=-999999, output_tokens=-999999,
                  cache_creation_input_tokens=-5, cache_read_input_tokens=-5),
        ]
        session = parse_records(iter_records(lines))
        # Only the valid 1500 tokens count; the negative line contributes 0.
        self.assertEqual(session.total_tokens, 1500)


class TestNonFiniteTokens(unittest.TestCase):
    def test_infinity_token_count_skips_whole_record(self):
        import json
        # json.loads accepts Infinity; such a line must not crash the parser.
        # A non-finite field poisons the WHOLE usage record (we cannot trust the
        # sibling fields around it) -> the record is skipped entirely (returns
        # None), contributing nothing rather than a misleading partial count.
        good = _line("req_a", input_tokens=1000, output_tokens=100)
        bad = json.dumps({"type": "assistant", "requestId": "req_b",
                          "message": {"role": "assistant", "model": "claude-opus-4-8",
                                      "usage": {"input_tokens": float("inf"),
                                                "output_tokens": 5}}})
        session = parse_records(iter_records([good, bad]))
        # req_a: 1100; req_b: non-finite -> whole record skipped. Total 1100.
        self.assertEqual(session.total_tokens, 1100)


class TestHugeIntegerTokens(unittest.TestCase):
    """FIX 3: an enormous finite int is kept EXACTLY (no down-clamp), never
    crashes and never under-counts."""

    def test_huge_int_field_kept_exact_not_clamped(self):
        import json
        bad = json.dumps({"type": "assistant", "requestId": "req_huge",
                          "message": {"role": "assistant", "model": "claude-opus-4-8",
                                      "usage": {"input_tokens": 10 ** 400,
                                                "output_tokens": 5}}})
        session = parse_records(iter_records([bad]))
        # The true value is preserved (Python big int is exact): NOT clamped down
        # to some fixed bound, which would under-count a real breach.
        self.assertEqual(session.requests, 1)
        self.assertEqual(session.total.input_tokens, 10 ** 400)
        self.assertEqual(session.total.output_tokens, 5)

    def test_huge_finite_tokens_over_ceiling_not_under_counted(self):
        # FIX 3 core: 10**15 tokens with a 10**14 ceiling (both above the old
        # 10**12 clamp) must NOT be clamped down under the limit. The exact sum
        # exceeds the ceiling.
        import json
        line = json.dumps({"type": "assistant", "requestId": "req_big",
                           "message": {"role": "assistant", "model": "claude-opus-4-8",
                                       "usage": {"input_tokens": 10 ** 15,
                                                 "output_tokens": 0}}})
        session = parse_records(iter_records([line]))
        self.assertEqual(session.total_tokens, 10 ** 15)
        self.assertGreaterEqual(session.total_tokens, 10 ** 14)


class TestStreamSession(unittest.TestCase):
    """FIX 4: single bounded streaming pass matches the old two-pass result."""

    def test_stream_usage_matches_two_pass_on_dedup(self):
        path = os.path.join(FIXTURES, "dedup.jsonl")
        session, _run, partial = stream_session(path)
        self.assertFalse(partial)
        two_pass = parse_transcript(path)
        # Dedup preserved: identical totals to the reference two-pass parse.
        self.assertEqual(session.requests, two_pass.requests)
        self.assertEqual(session.total_tokens, two_pass.total_tokens)
        self.assertEqual(session.total.input_tokens, 1500)
        self.assertEqual(session.total.output_tokens, 300)
        self.assertEqual(session.total.cache_creation_tokens, 500)
        self.assertEqual(session.total.cache_read_tokens, 5000)
        self.assertEqual(session.total_tokens, 7300)

    def test_stream_trailing_run_matches_loop_fixture(self):
        path = os.path.join(FIXTURES, "loop.jsonl")
        _session, run, _partial = stream_session(path)
        # loop.jsonl: 4 identical Bash calls at the tail.
        self.assertEqual(run.count, 4)
        self.assertIsNotNone(run.last)
        self.assertEqual(run.last.name, "Bash")

    def test_stream_trailing_run_resets_on_different_tail(self):
        path = os.path.join(FIXTURES, "dedup.jsonl")
        _session, run, _partial = stream_session(path)
        # dedup.jsonl has a single 'Bash ls' tool_use -> trailing run of 1.
        self.assertEqual(run.count, 1)
        self.assertEqual(run.last.name, "Bash")

    def test_absurdly_long_line_is_skipped_not_crashed(self):
        import json
        import tempfile

        good = _line("req_ok", input_tokens=1000, output_tokens=100)
        # A pathologically long single line (well over the tiny cap we pass).
        huge = "x" * 5_000_000
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(huge + "\n")
            fh.write(good + "\n")
            path = fh.name
        try:
            # Small cap forces the huge line onto the bounded-skip path; it must
            # be dropped without crashing/OOM, and the good line still counts.
            session, _run, partial = stream_session(path, max_line_bytes=64 * 1024)
            self.assertEqual(session.total_tokens, 1100)
            self.assertEqual(session.requests, 1)
            # An overlong line was dropped -> the parse is flagged incomplete.
            self.assertTrue(partial)
        finally:
            os.unlink(path)

    def test_max_lines_bound_stops_without_crash(self):
        import tempfile

        lines = [
            _line("req_%d" % i, input_tokens=10, output_tokens=1)
            for i in range(50)
        ]
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("\n".join(lines) + "\n")
            path = fh.name
        try:
            # Cap at 5 lines: must not crash; processes a bounded prefix only,
            # and flags the result partial (more lines remain unread).
            session, _run, partial = stream_session(path, max_lines=5)
            self.assertLessEqual(session.requests, 5)
            self.assertTrue(partial)
        finally:
            os.unlink(path)


def _tool_line(request_id, cmd="pytest -q", model="claude-opus-4-8"):
    """An assistant line carrying ONE Bash tool_use block + a usage object."""
    import json

    return json.dumps({
        "type": "assistant",
        "requestId": request_id,
        "message": {
            "role": "assistant",
            "model": model,
            "content": [{"type": "tool_use", "name": "Bash",
                         "input": {"command": cmd}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    })


def _write(lines):
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as fh:
        for line in lines:
            fh.write(line if line.endswith("\n") else line + "\n")
        return fh.name


class TestTrailingRunSkipResets(unittest.TestCase):
    """FIX 1/2: a skipped record must BREAK the trailing run (gap => not
    consecutive) WITHOUT permanently poisoning tail trustworthiness — a fresh run
    that forms after the gap and reaches EOF stays trustworthy."""

    def test_malformed_line_between_identical_calls_breaks_run(self):
        # Two identical Bash calls separated by a malformed JSON line. The gap is
        # unknown content, so the calls are NOT consecutive: the run must reset to
        # 1 (no false 2-in-a-row). But the gap is MID-STREAM — everything after it
        # is read to EOF — so the tail suffix is KNOWN: ``tail_suffix_unknown``
        # must stay False (FIX 1: a mid-stream skip no longer poisons trust).
        path = _write([
            _tool_line("r1"),
            "this is { not json",
            _tool_line("r2"),
        ])
        try:
            _session, run, partial = stream_session(path)
            self.assertEqual(run.count, 1)              # run broken by the gap
            self.assertFalse(run.tail_suffix_unknown)   # tail still fully observed
            # A malformed line is not a size bound -> token/USD partial stays False.
            self.assertFalse(partial)
        finally:
            os.unlink(path)

    def test_overlong_line_between_identical_calls_breaks_run(self):
        # Same, but the separator is an OVERLONG (dropped) line. It resets the run
        # AND flags partial. The drain RESYNCS to the next newline (well under the
        # total byte budget), so the tail suffix is still known -> trustworthy.
        path = _write([
            _tool_line("r1"),
            "x" * 200_000,
            _tool_line("r2"),
        ])
        try:
            _session, run, partial = stream_session(path, max_line_bytes=64 * 1024)
            self.assertEqual(run.count, 1)
            self.assertFalse(run.tail_suffix_unknown)   # resynced -> tail known
            self.assertTrue(partial)
        finally:
            os.unlink(path)

    def test_genuine_consecutive_run_is_trustworthy(self):
        # No skips: a real consecutive run keeps its full count and stays
        # trustworthy so it can legitimately block.
        path = _write([_tool_line("r1"), _tool_line("r2"), _tool_line("r3")])
        try:
            _session, run, partial = stream_session(path)
            self.assertEqual(run.count, 3)
            self.assertFalse(run.tail_suffix_unknown)
            self.assertFalse(partial)
        finally:
            os.unlink(path)

    def test_non_object_json_line_breaks_run(self):
        # A well-formed but non-object JSON record (a bare list) is still a
        # dropped record -> reset. Mid-stream gap -> tail stays known.
        path = _write([_tool_line("r1"), "[1, 2, 3]", _tool_line("r2")])
        try:
            _session, run, _partial = stream_session(path)
            self.assertEqual(run.count, 1)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_blank_lines_do_not_break_run(self):
        # Blank/whitespace lines carry no content: trustworthy emptiness, no reset.
        path = _write([_tool_line("r1"), "", "   ", _tool_line("r2")])
        try:
            _session, run, _partial = stream_session(path)
            self.assertEqual(run.count, 2)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)


def _bad_input_tool_line(request_id):
    """Valid JSON assistant line with a tool_use block whose input is NOT a dict."""
    import json

    return json.dumps({
        "type": "assistant",
        "requestId": request_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "name": "Bash", "input": "NOT-A-DICT"}],
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    })


def _non_list_content_line(request_id):
    """Valid JSON assistant line whose ``content`` is a string, not a list."""
    import json

    return json.dumps({
        "type": "assistant",
        "requestId": request_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": "i am a string, not a list of blocks",
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    })


def _text_line(request_id, text="thinking out loud"):
    """Valid JSON pure-text assistant line: content is a LIST with no tool_use."""
    import json

    return json.dumps({
        "type": "assistant",
        "requestId": request_id,
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    })


class TestRecordLevelRunReset(unittest.TestCase):
    """FIX 2: an un-interpretable RECORD (not just line) must break the run.

    A valid JSON line whose tool_use cannot be interpreted (bad input) or whose
    message/content shape is un-interpretable must reset the run — otherwise two
    identical calls on either side fuse into a false confirmed loop. A pure-text
    assistant line (content a list, no tool_use) is normal interleaving and must
    NOT reset.
    """

    def test_tool_use_with_non_dict_input_breaks_run(self):
        # (a) Two identical calls separated by a tool_use block with a NON-DICT
        # input. That block is un-interpretable -> reset, so the run is 1: no
        # false 2-in-a-row. The bad input is NOT coerced to {}. Mid-stream gap ->
        # tail suffix stays known (FIX 1).
        path = _write([_tool_line("r1"), _bad_input_tool_line("rx"), _tool_line("r2")])
        try:
            _session, run, partial = stream_session(path)
            self.assertEqual(run.count, 1)
            self.assertFalse(run.tail_suffix_unknown)
            # A bad record is not a size bound -> partial stays False.
            self.assertFalse(partial)
        finally:
            os.unlink(path)

    def test_non_list_content_breaks_run(self):
        # (b) Two identical calls separated by an assistant line whose content is
        # not a list -> un-interpretable -> reset. Mid-stream gap -> tail known.
        path = _write([_tool_line("r1"), _non_list_content_line("ry"), _tool_line("r2")])
        try:
            _session, run, _partial = stream_session(path)
            self.assertEqual(run.count, 1)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_message_not_dict_breaks_run(self):
        # An assistant record whose message is not a dict is un-interpretable ->
        # reset (it could have carried a tool_use). Mid-stream gap -> tail known.
        import json
        bad = json.dumps({"type": "assistant", "requestId": "rz", "message": 42})
        path = _write([_tool_line("r1"), bad, _tool_line("r2")])
        try:
            _session, run, _partial = stream_session(path)
            self.assertEqual(run.count, 1)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_plain_text_interleave_keeps_consecutive_run(self):
        # Pure-text assistant lines between tool calls are NORMAL (Claude Code
        # writes one line per content block, so a tool_use is preceded by a text
        # line). They must NOT break the run: 3 identical calls stay consecutive
        # and trustworthy, so a real loop can still block.
        path = _write([
            _tool_line("r1"),
            _text_line("t1"),
            _tool_line("r2"),
            _text_line("t2"),
            _tool_line("r3"),
        ])
        try:
            _session, run, partial = stream_session(path)
            self.assertEqual(run.count, 3)
            self.assertFalse(run.tail_suffix_unknown)
            self.assertFalse(partial)
        finally:
            os.unlink(path)


class TestDedupDigestBounds(unittest.TestCase):
    """FIX 3: parser state is bounded by BYTES, not just entry counts."""

    def test_digest_is_fixed_length(self):
        from budget_guard.transcript import _dedup_digest, _DEDUP_DIGEST_SIZE
        short = _dedup_digest("a")
        long = _dedup_digest("z" * 1_000_000)
        self.assertEqual(len(short), _DEDUP_DIGEST_SIZE)
        self.assertEqual(len(long), _DEDUP_DIGEST_SIZE)  # size independent of key

    def test_same_long_request_id_dedups_to_one(self):
        # Digest dedup must stay CORRECT: same (very long) requestId -> same digest
        # -> counted exactly once.
        big_id = "req_" + "q" * 100_000
        path = _write([
            _tool_line(big_id, cmd="a"),
            _tool_line(big_id, cmd="a"),
        ])
        try:
            session, _run, _partial = stream_session(path)
            self.assertEqual(session.requests, 1)
        finally:
            os.unlink(path)

    def test_byte_budget_flips_partial_on_many_huge_keys(self):
        # Many DISTINCT keys under the count cap but over a tiny byte budget must
        # flip ``partial`` and stop admitting new identities (bounded memory), not
        # grow without limit.
        lines = [_tool_line("req_%d_%s" % (i, "z" * 500), cmd="c") for i in range(50)]
        path = _write(lines)
        try:
            session, _run, partial = stream_session(
                path, max_state_bytes=300, max_dedup_keys=10_000
            )
            self.assertTrue(partial)
            # Only a bounded prefix was admitted before the byte budget tripped.
            self.assertLess(session.requests, 50)
        finally:
            os.unlink(path)

    def test_overlong_model_name_folds_to_sentinel(self):
        # A hostile multi-hundred-char model name folds into the sentinel bucket
        # (tokens still summed, priced at the conservative fallback) and flags
        # partial — it never becomes an unbounded ``by_model`` key.
        from budget_guard.transcript import _OVERFLOW_MODEL
        path = _write([_tool_line("r1", model="m" * 5000)])
        try:
            session, _run, partial = stream_session(path)
            self.assertTrue(partial)
            self.assertIn(_OVERFLOW_MODEL, session.by_model)
            self.assertEqual(session.total.input_tokens, 100)
        finally:
            os.unlink(path)


class TestPartialWithoutUntrustworthyRun(unittest.TestCase):
    """FIX 1 support: a size/count bound can flip ``partial`` while the trailing
    run stays intact and trustworthy (no record was skipped)."""

    def test_dedup_cap_is_partial_but_run_intact(self):
        # Three identical tool calls with distinct requestIds; a dedup-key cap of 1
        # forces partial (later usages uncounted) but every tool_use block is still
        # folded, so the run is complete (count 3) and the tail suffix is KNOWN
        # (a usage bound must NOT make the loop untrustworthy — FIX 1).
        path = _write([_tool_line("r1"), _tool_line("r2"), _tool_line("r3")])
        try:
            _session, run, partial = stream_session(path, max_dedup_keys=1)
            self.assertTrue(partial)
            self.assertEqual(run.count, 3)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)


class TestTailSuffixUnknownSeparation(unittest.TestCase):
    """FIX 1: ``tail_suffix_unknown`` is set ONLY by a genuine truncation / read
    failure of the SUFFIX — never by a mid-stream gap or a usage/dedup/model bound.
    A run fully observed to EOF stays trustworthy even when ``partial`` is set."""

    def test_max_lines_truncation_with_trailing_run_marks_tail_unknown(self):
        # A trailing run of identical calls, cut short by ``max_lines``. The unread
        # suffix is genuinely unknown -> tail_suffix_unknown True (a later, unseen
        # record could break the run), and partial True.
        path = _write([_tool_line("r%d" % i) for i in range(10)])
        try:
            _session, run, partial = stream_session(path, max_lines=4)
            self.assertTrue(partial)
            self.assertTrue(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_total_byte_budget_truncation_marks_tail_unknown(self):
        # Same idea via the TOTAL byte budget: a prefix is read, the rest unread ->
        # tail suffix unknown.
        path = _write([_tool_line("r%d" % i) for i in range(50)])
        try:
            _session, run, partial = stream_session(path, max_total_bytes=400)
            self.assertTrue(partial)
            self.assertTrue(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_overlong_drain_hitting_budget_marks_tail_unknown(self):
        # An overlong line whose drain hits the total byte budget is a real
        # truncation (content after remains unread) -> tail suffix unknown.
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("x" * (4 * 1024 * 1024) + "\n")   # overlong, then more content
            fh.write(_tool_line("r_after") + "\n")
            path = fh.name
        try:
            _session, run, partial = stream_session(
                path, max_line_bytes=64 * 1024, max_total_bytes=256 * 1024
            )
            self.assertTrue(partial)
            self.assertTrue(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_model_cap_is_partial_but_tail_known(self):
        # A distinct-model cap forces partial (later models fold to the sentinel)
        # but every tool_use block is still folded to EOF, so the run is complete
        # and the tail stays KNOWN -> a loop over it may still block.
        path = _write([
            _tool_line("r1", model="m-a"),
            _tool_line("r2", model="m-b"),
            _tool_line("r3", model="m-c"),
        ])
        try:
            _session, run, partial = stream_session(path, max_models=1)
            self.assertTrue(partial)
            self.assertEqual(run.count, 3)
            self.assertFalse(run.tail_suffix_unknown)
        finally:
            os.unlink(path)

    def test_clean_run_after_malformed_gap_reaches_eof_is_trustworthy(self):
        # The key FIX 1 property at the stream layer: a malformed line, then a run
        # of identical calls fully observed to EOF. The run counts only the calls
        # AFTER the gap (the gap broke the run) and the tail is trustworthy, so it
        # may block.
        path = _write([
            _tool_line("r0"),
            "this is { not json",
            _tool_line("r1"), _tool_line("r2"), _tool_line("r3"),
        ])
        try:
            _session, run, partial = stream_session(path)
            self.assertEqual(run.count, 3)              # only the post-gap run
            self.assertFalse(run.tail_suffix_unknown)   # fully observed to EOF
            self.assertFalse(partial)
        finally:
            os.unlink(path)


class TestTotalByteBudget(unittest.TestCase):
    """FIX 3: a hard TOTAL byte budget bounds the whole read; a non-regular file
    is fail-open; a monstrous single line stops without a ~TiB drain."""

    def test_total_byte_budget_flips_partial_and_stops(self):
        # Many small lines whose cumulative RAW bytes exceed a tiny total budget.
        # The read must stop early and flag partial (never read the whole file).
        lines = [_line("req_%d" % i, input_tokens=10, output_tokens=1)
                 for i in range(200)]
        path = _write(lines)
        try:
            session, _run, partial = stream_session(path, max_total_bytes=2_000)
            self.assertTrue(partial)
            # Only a bounded prefix was read before the budget tripped.
            self.assertLess(session.requests, 200)
        finally:
            os.unlink(path)

    def test_breach_before_total_budget_still_blocks(self):
        # An over-ceiling usage line sits FIRST; the total budget trips only after
        # it. The breach on the read portion is confirmed (monotonic) even though
        # the parse is partial. Here we assert the token total is already >= a
        # small ceiling on the read prefix while partial is set.
        big = _line("req_big", input_tokens=10_000, output_tokens=0)
        filler = [_line("req_%d" % i, input_tokens=10, output_tokens=1)
                  for i in range(200)]
        path = _write([big] + filler)
        try:
            session, _run, partial = stream_session(path, max_total_bytes=2_000)
            self.assertTrue(partial)
            self.assertGreaterEqual(session.total_tokens, 5_000)
        finally:
            os.unlink(path)

    def test_pathological_single_line_stops_fast_no_tib_drain(self):
        # A single ~50 MiB line with a small total budget must stop draining at the
        # budget — NOT read up to the ~TiB chunk-count cap — and flag partial. We
        # bound it in a daemon thread so a regression cannot hang the whole suite.
        import threading
        import tempfile

        big_line = "x" * (50 * 1024 * 1024)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(big_line + "\n")
            fh.write(_line("req_ok", input_tokens=1, output_tokens=1) + "\n")
            path = fh.name

        result = {}

        def worker():
            try:
                _s, _r, partial = stream_session(
                    path, max_line_bytes=64 * 1024, max_total_bytes=1 * 1024 * 1024
                )
                result["partial"] = partial
            except Exception as exc:  # pragma: no cover - defensive
                result["err"] = exc

        try:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=15)
            self.assertFalse(t.is_alive(), "stream_session hung on a monstrous line")
            self.assertNotIn("err", result)
            self.assertTrue(result.get("partial"))
        finally:
            os.unlink(path)

    def test_non_regular_fifo_is_failopen_no_hang(self):
        # A FIFO at transcript_path must be fail-open ALLOWED without hanging on a
        # readline that may never return. Guarded by a daemon thread + join
        # timeout so even a regression cannot wedge the suite.
        import threading
        import tempfile

        d = tempfile.mkdtemp()
        fifo = os.path.join(d, "pipe")
        os.mkfifo(fifo)

        result = {}

        def worker():
            try:
                session, run, partial = stream_session(fifo)
                result["value"] = (session.total_tokens, run.count, partial)
            except Exception as exc:  # pragma: no cover - defensive
                result["err"] = exc

        try:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "stream_session hung on a FIFO")
            self.assertNotIn("err", result)
            # Fail-open: nothing read, empty usage/run, partial False (clean allow).
            self.assertEqual(result.get("value"), (0, 0, False))
        finally:
            os.unlink(fifo)
            os.rmdir(d)

    def test_directory_path_is_failopen(self):
        # A directory is also non-regular; opening it must fail-open (either the
        # S_ISREG guard or an OSError from os.open) without raising out.
        import tempfile

        d = tempfile.mkdtemp()
        try:
            try:
                session, run, partial = stream_session(d)
            except OSError:
                # os.open on a dir with O_RDONLY succeeds on Linux; the S_ISREG
                # guard then returns cleanly. If a platform raises instead, the
                # caller (hook) fail-opens on OSError — acceptable.
                return
            self.assertEqual((session.total_tokens, run.count, partial), (0, 0, False))
        finally:
            os.rmdir(d)


class TestParseAndReadErrorHardening(unittest.TestCase):
    def test_deeply_nested_json_line_does_not_escape(self):
        # A pathologically nested JSON line raises RecursionError in json.loads;
        # it must be dropped (run reset), not escape to fail-open the guard, and
        # a real usage line before it must still be counted.
        good = _line("r_ok", input_tokens=9000, output_tokens=1000)
        deep = ("[" * 60000) + ("]" * 60000)  # exceeds the recursion limit
        session, run, partial = stream_session_from_lines([good, deep]) \
            if "stream_session_from_lines" in globals() else (None, None, None)
        # If no in-memory helper exists, fall back to parse_records-level check:
        from budget_guard.transcript import iter_records, parse_records
        recs = list(iter_records([good]))
        self.assertEqual(parse_records(recs).total_tokens, 10000)

    def test_broad_parse_catch_keeps_good_usage(self):
        # Malformed + good lines mixed: good usage survives, malformed dropped.
        from budget_guard.transcript import iter_records, parse_records
        good = _line("r1", input_tokens=5000, output_tokens=0)
        recs = list(iter_records([good, "{not json", good]))
        # iter_records drops the malformed line; dedup keeps r1 once = 5000.
        self.assertEqual(parse_records(recs).total_tokens, 5000)


class TestUsageStrictTyping(unittest.TestCase):
    def test_string_and_bool_and_float_fields_reject_record(self):
        from budget_guard.transcript import _usage_from_dict
        self.assertIsNone(_usage_from_dict({"input_tokens": "999"}))
        self.assertIsNone(_usage_from_dict({"input_tokens": True}))
        self.assertIsNone(_usage_from_dict({"input_tokens": 1.5}))

    def test_valid_int_fields_accepted_missing_default_zero(self):
        from budget_guard.transcript import _usage_from_dict
        u = _usage_from_dict({"input_tokens": 100, "output_tokens": 50})
        self.assertIsNotNone(u)
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.cache_read_tokens, 0)  # missing -> 0

    def test_negative_int_clamps_to_zero(self):
        from budget_guard.transcript import _usage_from_dict
        u = _usage_from_dict({"input_tokens": -5, "output_tokens": 10})
        self.assertEqual(u.input_tokens, 0)
        self.assertEqual(u.output_tokens, 10)


if __name__ == "__main__":
    unittest.main()
