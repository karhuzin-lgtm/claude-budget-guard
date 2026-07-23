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


def payload_for(path):
    return {
        "session_id": "s1",
        "transcript_path": path,
        "cwd": "/tmp",
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "echo hi"},
    }


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
        # loop.jsonl has 4 identical Bash calls.
        code, _out, err = invoke(payload_for(LOOP), CLAUDE_BUDGET_LOOP_LIMIT=4)
        self.assertEqual(code, EXIT_BLOCK)
        self.assertIn("Retry-loop detected", err)

    def test_loop_below_limit_allows(self):
        code, _out, _err = invoke(payload_for(LOOP), CLAUDE_BUDGET_LOOP_LIMIT=10)
        self.assertEqual(code, EXIT_ALLOW)


class TestJsonOutputMode(unittest.TestCase):
    def test_block_as_json_stdout_exit0(self):
        code, out, err = invoke(
            payload_for(DEDUP),
            CLAUDE_BUDGET_MAX_TOKENS=5000,
            CLAUDE_BUDGET_OUTPUT="json",
        )
        self.assertEqual(code, EXIT_ALLOW)  # json mode never exits 2
        self.assertEqual(err, "")
        data = json.loads(out)
        self.assertEqual(data["decision"], "block")
        self.assertIn("Token budget exceeded", data["reason"])


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


if __name__ == "__main__":
    unittest.main()
