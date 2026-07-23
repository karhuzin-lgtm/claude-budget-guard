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


class TestCostOverflow(unittest.TestCase):
    """FIX A: an un-priceable (astronomical) token count reads as +inf."""

    def test_huge_int_cost_is_inf_not_overflow(self):
        price = ModelPrice(input=15.0, output=75.0, cache_write=1.0, cache_read=1.0)
        # 10**400 * 15.0 raises OverflowError normally; must degrade to +inf.
        cost = price.cost(Usage(input_tokens=10 ** 400))
        self.assertEqual(cost, float("inf"))

    def test_session_cost_with_overflow_is_inf(self):
        session = SessionUsage()
        session.add("claude-opus-4-8", Usage(input_tokens=10 ** 400))
        table = PricingTable()
        self.assertEqual(table.session_cost(session), float("inf"))


class TestLongestMatchAndOverride(unittest.TestCase):
    """FIX 2: most-specific (longest) match wins; case-insensitive replace."""

    def test_custom_specific_family_beats_generic_default(self):
        # "claude-opus-4-8" (custom) must win over the generic "opus" default
        # for model "claude-opus-4-8" — longest matching key wins.
        table = PricingTable.from_mapping(
            {"claude-opus-4-8": {"input": 1.0, "output": 2.0,
                                 "cache_write": 3.0, "cache_read": 4.0}}
        )
        price = table.rate_for("claude-opus-4-8")
        self.assertEqual(price.input, 1.0)
        self.assertEqual(price.output, 2.0)

    def test_uppercase_key_overrides_lowercase_default(self):
        # "OPUS" must replace the "opus" default, not create a stale duplicate.
        table = PricingTable.from_mapping(
            {"OPUS": {"input": 42.0, "output": 43.0,
                      "cache_write": 1.0, "cache_read": 1.0}}
        )
        self.assertEqual(table.rate_for("claude-opus-4-8").input, 42.0)
        # No stale/duplicate: exactly one opus-family key remains.
        self.assertIn("opus", table.rates)
        self.assertNotIn("OPUS", table.rates)
        self.assertEqual(
            sum(1 for k in table.rates if k.lower() == "opus"), 1
        )

    def test_longest_match_wins_when_two_families_match(self):
        # Both "opus" and "opus-4" are substrings of the model; the longer,
        # more specific "opus-4" must win.
        table = PricingTable.from_mapping(
            {"opus-4": {"input": 7.0, "output": 8.0,
                        "cache_write": 1.0, "cache_read": 1.0}}
        )
        price = table.rate_for("claude-opus-4-8")
        self.assertEqual(price.input, 7.0)
        # A model matching only the generic "opus" still uses the default.
        self.assertEqual(table.rate_for("claude-opus-3").input, 15.0)


class TestIncompletePricingRejected(unittest.TestCase):
    """FIX 3: entries missing mandatory input/output are rejected, not zeroed."""

    def test_typo_output_field_rejected_keeps_default(self):
        from budget_guard.pricing import DEFAULT_RATES
        # 'ouput' typo => 'output' missing => whole entry rejected; opus keeps
        # its DEFAULT output rate (75.0), NOT a silently-zeroed 0.0.
        table = PricingTable.from_mapping({"opus": {"input": 15, "ouput": 75}})
        price = table.rate_for("claude-opus-4-8")
        self.assertEqual(price.output, DEFAULT_RATES["opus"].output)
        self.assertEqual(price.output, 75.0)
        # And input stays the default too (entry rejected wholesale).
        self.assertEqual(price.input, DEFAULT_RATES["opus"].input)

    def test_missing_input_rejected(self):
        from budget_guard.pricing import DEFAULT_RATES
        table = PricingTable.from_mapping({"opus": {"output": 99.0}})
        self.assertEqual(table.rate_for("claude-opus-4-8").input,
                         DEFAULT_RATES["opus"].input)
        self.assertEqual(table.rate_for("claude-opus-4-8").output,
                         DEFAULT_RATES["opus"].output)

    def test_input_output_present_cache_inherits_from_input(self):
        # A complete entry (input+output) without cache_* is ACCEPTED and cache
        # rates intentionally inherit the input rate.
        table = PricingTable.from_mapping({"opus": {"input": 10.0, "output": 20.0}})
        price = table.rate_for("claude-opus-4-8")
        self.assertEqual(price.input, 10.0)
        self.assertEqual(price.output, 20.0)
        self.assertEqual(price.cache_write, 10.0)
        self.assertEqual(price.cache_read, 10.0)


class TestEmptyFamilyKey(unittest.TestCase):
    def test_empty_family_key_rejected(self):
        from budget_guard.pricing import PricingTable, DEFAULT_FALLBACK
        table = PricingTable.from_mapping({"": {"input": 0.0, "output": 0.0,
                                                "cache_write": 0.0, "cache_read": 0.0}})
        # Empty key must be ignored -> unknown models still use the fallback rate,
        # not a zeroed cost.
        self.assertEqual(table.rate_for("some-unknown-model").input,
                         DEFAULT_FALLBACK.input)


class TestPricingOverflowIntIgnored(unittest.TestCase):
    def test_enormous_int_rate_ignored_not_raised(self):
        from budget_guard.pricing import PricingTable, DEFAULT_RATES
        # A JSON int too big for float() must not raise out of from_mapping
        # (which would abort and, in the hook, disable even the token limit).
        table = PricingTable.from_mapping(
            {"opus": {"input": 10**400, "output": 75,
                      "cache_write": 1.0, "cache_read": 1.0}})
        # bad entry ignored -> opus keeps its safe default input rate
        self.assertEqual(table.rate_for("claude-opus-4-8").input,
                         DEFAULT_RATES["opus"].input)


if __name__ == "__main__":
    unittest.main()
