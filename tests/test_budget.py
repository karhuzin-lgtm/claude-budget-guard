"""Tests for the allow/warn/block decision logic."""

from __future__ import annotations

import unittest

from budget_guard.budget import ALLOW, BLOCK, WARN, decide
from budget_guard.config import Config
from budget_guard.loops import LoopInfo


class TestTokenCeiling(unittest.TestCase):
    def test_under_budget_allows(self):
        cfg = Config(max_tokens=1000, warn_pct=80)
        d = decide(tokens=500, usd=0.0, config=cfg)
        self.assertEqual(d.action, ALLOW)

    def test_warn_threshold(self):
        cfg = Config(max_tokens=1000, warn_pct=80)
        d = decide(tokens=850, usd=0.0, config=cfg)
        self.assertEqual(d.action, WARN)
        self.assertIn("%", d.message)

    def test_at_limit_blocks(self):
        cfg = Config(max_tokens=1000, warn_pct=80)
        d = decide(tokens=1000, usd=0.0, config=cfg)
        self.assertEqual(d.action, BLOCK)
        self.assertTrue(d.blocked)
        self.assertIn("Token budget exceeded", d.message)

    def test_over_limit_blocks(self):
        cfg = Config(max_tokens=1000)
        d = decide(tokens=5000, usd=0.0, config=cfg)
        self.assertEqual(d.action, BLOCK)

    def test_no_limit_allows_huge_usage(self):
        cfg = Config()  # zero-config
        d = decide(tokens=10_000_000, usd=0.0, config=cfg)
        self.assertEqual(d.action, ALLOW)


class TestUsdCeiling(unittest.TestCase):
    def test_usd_warn(self):
        cfg = Config(max_usd=10.0, warn_pct=80)
        d = decide(tokens=0, usd=9.0, config=cfg)
        self.assertEqual(d.action, WARN)

    def test_usd_block(self):
        cfg = Config(max_usd=10.0)
        d = decide(tokens=0, usd=12.0, config=cfg)
        self.assertEqual(d.action, BLOCK)
        self.assertIn("USD budget exceeded", d.message)


class TestLoop(unittest.TestCase):
    def test_loop_blocks(self):
        cfg = Config(loop_limit=5)
        loop = LoopInfo(name="Bash", count=5, signature="sig")
        d = decide(tokens=10, usd=0.0, config=cfg, loop=loop)
        self.assertEqual(d.action, BLOCK)
        self.assertIn("Retry-loop detected", d.message)

    def test_block_priority_over_warn(self):
        # Token warn but a loop block -> overall block.
        cfg = Config(max_tokens=1000, warn_pct=80, loop_limit=3)
        loop = LoopInfo(name="Read", count=3, signature="sig")
        d = decide(tokens=850, usd=0.0, config=cfg, loop=loop)
        self.assertEqual(d.action, BLOCK)


class TestLoopTrustworthiness(unittest.TestCase):
    """FIX 1: a loop may block only on a trustworthy read; otherwise it warns."""

    def test_untrustworthy_loop_warns_not_blocks(self):
        # Same loop, but the read was partial/untrustworthy -> unproven -> WARN.
        cfg = Config(loop_limit=5)
        loop = LoopInfo(name="Bash", count=9, signature="sig")
        d = decide(tokens=10, usd=0.0, config=cfg, loop=loop, loop_trustworthy=False)
        self.assertEqual(d.action, WARN)
        self.assertNotEqual(d.action, BLOCK)
        self.assertIn("Possible retry-loop", d.message)

    def test_trustworthy_loop_still_blocks(self):
        cfg = Config(loop_limit=5)
        loop = LoopInfo(name="Bash", count=5, signature="sig")
        d = decide(tokens=10, usd=0.0, config=cfg, loop=loop, loop_trustworthy=True)
        self.assertEqual(d.action, BLOCK)
        self.assertIn("Retry-loop detected", d.message)

    def test_token_breach_still_blocks_under_untrustworthy_loop(self):
        # Token/USD are monotonic: a confirmed token breach blocks even when the
        # co-occurring loop is unproven. The loop must not be the blocking reason.
        cfg = Config(max_tokens=1000, loop_limit=5)
        loop = LoopInfo(name="Bash", count=9, signature="sig")
        d = decide(tokens=5000, usd=0.0, config=cfg, loop=loop, loop_trustworthy=False)
        self.assertEqual(d.action, BLOCK)
        self.assertIn("Token budget exceeded", d.message)
        self.assertNotIn("Retry-loop detected", d.message)


class TestMultipleReasons(unittest.TestCase):
    def test_token_and_usd_both_over(self):
        cfg = Config(max_tokens=1000, max_usd=10.0)
        d = decide(tokens=2000, usd=20.0, config=cfg)
        self.assertEqual(d.action, BLOCK)
        self.assertEqual(len(d.reasons), 2)


class TestConfigNonFiniteLimit(unittest.TestCase):
    def test_infinite_usd_limit_is_not_active(self):
        from budget_guard.config import load_config
        cfg = load_config(env={"CLAUDE_BUDGET_MAX_USD": "inf"})
        # An infinite USD limit is meaningless; it must NOT register as active.
        self.assertIsNone(cfg.max_usd)
        self.assertFalse(cfg.has_any_limit)


class TestHugeTokenFormatting(unittest.TestCase):
    def test_absurd_token_count_does_not_raise_on_format(self):
        # ~10000-digit token count exceeds Python's int->str limit; decide() must
        # still render a block reason, not raise into the hook's fail-open path.
        from budget_guard.budget import decide
        from budget_guard.config import Config
        huge = 10**10000
        d = decide(tokens=huge, usd=0.0, config=Config(max_tokens=10**9))
        self.assertEqual(d.action, "block")
        self.assertTrue(d.message)  # renders without raising


class TestConfigFileSafety(unittest.TestCase):
    def test_oversized_config_file_ignored(self):
        import os, tempfile
        from budget_guard.config import _load_file, MAX_CONFIG_BYTES
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{" + '"x":' + "1," * (MAX_CONFIG_BYTES // 4) + '"max_tokens":5}')
            name = f.name
        try:
            self.assertEqual(_load_file(name), {})  # oversized -> defaults
        finally:
            os.unlink(name)

    def test_nonregular_config_path_ignored(self):
        import os, tempfile
        from budget_guard.config import _load_file
        d = tempfile.mkdtemp()
        fifo = os.path.join(d, "cfg.fifo")
        try:
            os.mkfifo(fifo)
            self.assertEqual(_load_file(fifo), {})  # FIFO -> ignored, no hang
        finally:
            try: os.remove(fifo)
            except OSError: pass
            os.rmdir(d)


if __name__ == "__main__":
    unittest.main()
