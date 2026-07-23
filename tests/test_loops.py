"""Tests for retry-loop detection."""

from __future__ import annotations

import unittest

from budget_guard.loops import detect_loop, detect_loop_run, trailing_repeat
from budget_guard.transcript import ToolCall, TrailingRun


def _calls(*specs):
    return [ToolCall(name=n, input=i) for n, i in specs]


class TestTrailingRepeat(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(trailing_repeat([]), 0)

    def test_single(self):
        self.assertEqual(trailing_repeat(_calls(("Bash", {"command": "ls"}))), 1)

    def test_all_identical(self):
        calls = _calls(*[("Bash", {"command": "ls"})] * 5)
        self.assertEqual(trailing_repeat(calls), 5)

    def test_trailing_run_only(self):
        # An earlier different call breaks the run; only the tail counts.
        calls = _calls(
            ("Read", {"file": "a"}),
            ("Bash", {"command": "ls"}),
            ("Bash", {"command": "ls"}),
            ("Bash", {"command": "ls"}),
        )
        self.assertEqual(trailing_repeat(calls), 3)

    def test_different_input_breaks_run(self):
        calls = _calls(
            ("Bash", {"command": "ls"}),
            ("Bash", {"command": "pwd"}),  # different input
        )
        self.assertEqual(trailing_repeat(calls), 1)

    def test_key_order_insensitive(self):
        calls = _calls(
            ("Bash", {"a": 1, "b": 2}),
            ("Bash", {"b": 2, "a": 1}),
        )
        self.assertEqual(trailing_repeat(calls), 2)


class TestDetectLoop(unittest.TestCase):
    def test_disabled_when_limit_none(self):
        calls = _calls(*[("Bash", {"command": "ls"})] * 20)
        self.assertIsNone(detect_loop(calls, None))

    def test_disabled_when_limit_zero(self):
        calls = _calls(*[("Bash", {"command": "ls"})] * 20)
        self.assertIsNone(detect_loop(calls, 0))

    def test_below_limit_none(self):
        calls = _calls(*[("Bash", {"command": "ls"})] * 3)
        self.assertIsNone(detect_loop(calls, 5))

    def test_at_limit_detected(self):
        calls = _calls(*[("Bash", {"command": "ls"})] * 5)
        info = detect_loop(calls, 5)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "Bash")
        self.assertEqual(info.count, 5)

    def test_above_limit_detected(self):
        calls = _calls(*[("Read", {"file": "x"})] * 12)
        info = detect_loop(calls, 10)
        self.assertIsNotNone(info)
        self.assertEqual(info.count, 12)

    def test_empty_calls(self):
        self.assertIsNone(detect_loop([], 5))


class TestDetectLoopRun(unittest.TestCase):
    """Streaming trailing-run loop check (used by the hook hot path)."""

    def _run(self, *specs):
        run = TrailingRun()
        for n, i in specs:
            run.add(ToolCall(name=n, input=i))
        return run

    def test_empty_run_none(self):
        self.assertIsNone(detect_loop_run(TrailingRun(), 5))

    def test_disabled_when_limit_none_or_zero(self):
        run = self._run(*[("Bash", {"command": "ls"})] * 20)
        self.assertIsNone(detect_loop_run(run, None))
        self.assertIsNone(detect_loop_run(run, 0))

    def test_below_limit_none(self):
        run = self._run(*[("Bash", {"command": "ls"})] * 3)
        self.assertIsNone(detect_loop_run(run, 5))

    def test_at_limit_detected(self):
        run = self._run(*[("Bash", {"command": "ls"})] * 5)
        info = detect_loop_run(run, 5)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "Bash")
        self.assertEqual(info.count, 5)

    def test_different_tail_resets_run(self):
        run = self._run(
            ("Bash", {"command": "ls"}),
            ("Bash", {"command": "ls"}),
            ("Read", {"file": "x"}),  # breaks the run
        )
        self.assertEqual(run.count, 1)
        self.assertIsNone(detect_loop_run(run, 2))

    def test_equivalent_to_detect_loop_on_same_sequence(self):
        specs = [("Bash", {"command": "ls"})] * 6
        run = self._run(*specs)
        calls = [ToolCall(name=n, input=i) for n, i in specs]
        a = detect_loop_run(run, 5)
        b = detect_loop(calls, 5)
        self.assertEqual((a.name, a.count), (b.name, b.count))
        self.assertEqual(a.signature, b.signature)


if __name__ == "__main__":
    unittest.main()
