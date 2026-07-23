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
    def test_infinity_token_count_does_not_crash_and_reads_zero(self):
        import json
        # json.loads accepts Infinity; such a line must not crash the parser
        # nor inflate/deflate the total — it reads as 0.
        good = _line("req_a", input_tokens=1000, output_tokens=100)
        bad = json.dumps({"type": "assistant", "requestId": "req_b",
                          "message": {"role": "assistant", "model": "claude-opus-4-8",
                                      "usage": {"input_tokens": float("inf"),
                                                "output_tokens": 5}}})
        session = parse_records(iter_records([good, bad]))
        # req_a: 1100; req_b: inf->0 input + 5 output = 5. Total 1105.
        self.assertEqual(session.total_tokens, 1105)


if __name__ == "__main__":
    unittest.main()
