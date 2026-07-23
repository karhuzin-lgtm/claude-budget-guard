"""Tests for the configurable pricing layer."""

from __future__ import annotations

import unittest

from budget_guard.pricing import ModelPrice, PricingTable
from budget_guard.transcript import SessionUsage, Usage


class TestModelPrice(unittest.TestCase):
    def test_cost_math_per_million(self):
        price = ModelPrice(input=15.0, output=75.0, cache_write=18.75, cache_read=1.5)
        usage = Usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        # 15 + 75 + 18.75 + 1.5
        self.assertAlmostEqual(price.cost(usage), 110.25)

    def test_fractional_tokens(self):
        price = ModelPrice(input=3.0, output=15.0, cache_write=3.75, cache_read=0.3)
        usage = Usage(input_tokens=500_000, output_tokens=100_000)
        # 0.5*3 + 0.1*15 = 1.5 + 1.5
        self.assertAlmostEqual(price.cost(usage), 3.0)


class TestPricingTable(unittest.TestCase):
    def test_substring_match(self):
        table = PricingTable()
        self.assertEqual(table.rate_for("claude-opus-4-8"), table.rates["opus"])
        self.assertEqual(table.rate_for("claude-3-5-sonnet"), table.rates["sonnet"])
        self.assertEqual(table.rate_for("claude-haiku-4-5"), table.rates["haiku"])

    def test_unknown_model_uses_fallback(self):
        table = PricingTable()
        self.assertEqual(table.rate_for("some-future-model"), table.fallback)

    def test_session_cost_sums_across_models(self):
        session = SessionUsage()
        session.add("claude-opus-4-8", Usage(input_tokens=1_000_000))   # 15.0
        session.add("claude-haiku-4-5", Usage(output_tokens=1_000_000))  # 4.0
        table = PricingTable()
        self.assertAlmostEqual(table.session_cost(session), 19.0)

    def test_from_mapping_override(self):
        table = PricingTable.from_mapping(
            {"opus": {"input": 1, "output": 2, "cache_write": 3, "cache_read": 4}}
        )
        price = table.rate_for("claude-opus-4-8")
        self.assertEqual(
            (price.input, price.output, price.cache_write, price.cache_read),
            (1.0, 2.0, 3.0, 4.0),
        )
        # Non-overridden families keep defaults.
        self.assertEqual(table.rate_for("sonnet").input, 3.0)

    def test_from_mapping_ignores_malformed(self):
        table = PricingTable.from_mapping(
            {"opus": "not-a-dict", 123: {"input": 1}, "haiku": {"input": "x"}}
        )
        # Falls back to defaults everywhere; must not raise.
        self.assertEqual(table.rate_for("opus").output, 75.0)

    def test_from_mapping_none(self):
        table = PricingTable.from_mapping(None)
        self.assertEqual(table.rate_for("opus").input, 15.0)


class TestPricingHardening(unittest.TestCase):
    def test_negative_and_nonfinite_rates_rejected(self):
        from budget_guard.pricing import PricingTable, DEFAULT_RATES
        bad = {
            "opus": {"input": -100.0, "output": -100.0,
                     "cache_write": -1.0, "cache_read": -1.0},
            "sonnet": {"input": float("inf"), "output": 1.0,
                       "cache_write": 1.0, "cache_read": 1.0},
            "haiku": {"input": float("nan"), "output": 1.0,
                      "cache_write": 1.0, "cache_read": 1.0},
        }
        table = PricingTable.from_mapping(bad)
        # All three malformed specs ignored -> safe DEFAULT_RATES kept.
        self.assertEqual(table.rate_for("claude-opus-4-8").input,
                         DEFAULT_RATES["opus"].input)
        self.assertEqual(table.rate_for("claude-sonnet-5").input,
                         DEFAULT_RATES["sonnet"].input)

    def test_valid_custom_rates_still_applied(self):
        from budget_guard.pricing import PricingTable
        table = PricingTable.from_mapping(
            {"opus": {"input": 99.0, "output": 1.0,
                      "cache_write": 1.0, "cache_read": 1.0}})
        self.assertEqual(table.rate_for("claude-opus-4-8").input, 99.0)


class TestEmptyFamilyKey(unittest.TestCase):
    def test_empty_family_key_rejected(self):
        from budget_guard.pricing import PricingTable, DEFAULT_FALLBACK
        table = PricingTable.from_mapping({"": {"input": 0.0, "output": 0.0,
                                                "cache_write": 0.0, "cache_read": 0.0}})
        # Empty key must be ignored -> unknown models still use the fallback rate,
        # not a zeroed cost.
        self.assertEqual(table.rate_for("some-unknown-model").input,
                         DEFAULT_FALLBACK.input)


if __name__ == "__main__":
    unittest.main()
