"""Convert token usage to an estimated USD cost.

Tokens are the *primary* budget metric everywhere else in this tool because
they are deterministic and never go stale. USD is a convenience layer on top,
and it is only as trustworthy as the rates you feed it.

The default rates below are **example figures** for illustration. Anthropic
pricing changes; do not treat these numbers as authoritative. If you set a USD
budget you should override the rates for your models via ``CLAUDE_BUDGET_CONFIG``
(see ``config``) so the estimate reflects reality. Rates are expressed per
*million* tokens, split by billing bucket: input, output, cache-write (a.k.a.
cache-creation) and cache-read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .transcript import SessionUsage, Usage


@dataclass
class ModelPrice:
    """Per-million-token USD rates for one model family."""

    input: float
    output: float
    cache_write: float
    cache_read: float

    def cost(self, usage: Usage) -> float:
        million = 1_000_000.0
        try:
            return (
                usage.input_tokens * self.input
                + usage.output_tokens * self.output
                + usage.cache_creation_tokens * self.cache_write
                + usage.cache_read_tokens * self.cache_read
            ) / million
        except (OverflowError, ValueError):
            # An un-priceable (astronomically large) token count multiplied by a
            # float raises OverflowError ("int too large to convert to float").
            # Read that as +inf: "definitely over any finite USD ceiling", never
            # as a swallowed error that would fail-open past the budget.
            return float("inf")


# EXAMPLE DEFAULTS — verify current Anthropic pricing before relying on USD
# limits. Keys are matched as case-insensitive *substrings* of the model name
# (e.g. "claude-opus-4-8" matches "opus"). Override via CLAUDE_BUDGET_CONFIG.
DEFAULT_RATES: Dict[str, ModelPrice] = {
    "opus": ModelPrice(input=15.0, output=75.0, cache_write=18.75, cache_read=1.50),
    "sonnet": ModelPrice(input=3.0, output=15.0, cache_write=3.75, cache_read=0.30),
    "haiku": ModelPrice(input=0.80, output=4.0, cache_write=1.0, cache_read=0.08),
}

# Fallback when a model name matches no known family. Uses the most expensive
# defaults (opus-like) so a USD ceiling errs on the safe/conservative side
# rather than silently under-counting an unknown model.
DEFAULT_FALLBACK = ModelPrice(
    input=15.0, output=75.0, cache_write=18.75, cache_read=1.50
)


class PricingTable:
    """Maps model names to rates and computes session cost."""

    def __init__(
        self,
        rates: Optional[Dict[str, ModelPrice]] = None,
        fallback: Optional[ModelPrice] = None,
    ) -> None:
        # Normalize family keys to lowercase so matching is case-insensitive and
        # a custom "OPUS" replaces the default "opus" instead of leaving a stale
        # duplicate (later insertion wins). Non-str / empty keys are dropped.
        source = DEFAULT_RATES if rates is None else rates
        normalized: Dict[str, ModelPrice] = {}
        for family, price in source.items():
            if not isinstance(family, str):
                continue
            key = family.strip().lower()
            if not key:
                continue
            normalized[key] = price
        self.rates = normalized
        self.fallback = DEFAULT_FALLBACK if fallback is None else fallback

    def rate_for(self, model: str) -> ModelPrice:
        name = (model or "").lower()
        # Choose the MOST SPECIFIC match: the LONGEST family key that is a
        # substring of the model name. This lets a custom "claude-opus-4-8" beat
        # the generic "opus" default regardless of insertion order (first-match
        # would wrongly return whichever was inserted first, undercounting cost).
        best: Optional[ModelPrice] = None
        best_len = -1
        for family, price in self.rates.items():
            if family in name and len(family) > best_len:
                best = price
                best_len = len(family)
        return best if best is not None else self.fallback

    def cost_for_model(self, model: str, usage: Usage) -> float:
        return self.rate_for(model).cost(usage)

    def session_cost(self, session: SessionUsage) -> float:
        """Total estimated USD across every model used in the session."""
        return sum(
            self.cost_for_model(model, usage)
            for model, usage in session.by_model.items()
        )

    @classmethod
    def from_mapping(cls, mapping: Optional[dict]) -> "PricingTable":
        """Build a table from a JSON-style dict.

        ``mapping`` shape::

            {"opus": {"input": 15, "output": 75,
                      "cache_write": 18.75, "cache_read": 1.5}, ...}

        Unknown families are added; known ones are overridden. Malformed entries
        are ignored so a bad config never crashes the fail-open hook.
        """
        if not isinstance(mapping, dict):
            return cls()
        rates = dict(DEFAULT_RATES)
        for family, spec in mapping.items():
            if not isinstance(family, str) or not isinstance(spec, dict):
                continue
            # An empty/whitespace family key is a substring of EVERY model name
            # ("" in name is always True), so it would hijack pricing for all
            # unknown models — potentially zeroing cost. Reject it. Lowercase so a
            # custom "OPUS" overrides the default "opus" (no stale duplicate).
            family = family.strip().lower()
            if not family:
                continue
            # The mandatory rate fields are ``input`` AND ``output``. If either is
            # MISSING, reject the whole entry (keep the safe existing/default rate)
            # rather than silently defaulting it to 0 and under-counting cost — a
            # typo like {"input":15,"ouput":75} must not zero the output rate.
            # ``cache_write`` / ``cache_read`` MAY still fall back to ``input``
            # when absent: that inheritance is intentional, unlike the mandatory
            # fields, because cache rates commonly track the input rate.
            if "input" not in spec or "output" not in spec:
                continue
            try:
                candidate = ModelPrice(
                    input=float(spec["input"]),
                    output=float(spec["output"]),
                    cache_write=float(spec.get("cache_write", spec["input"])),
                    cache_read=float(spec.get("cache_read", spec["input"])),
                )
            except (TypeError, ValueError, OverflowError):
                # OverflowError: float() of an enormous JSON int. Ignore the bad
                # entry (keep the safe default) — a pricing typo must never abort
                # from_mapping and thereby disable the independent token limit.
                continue
            # Reject non-finite (nan/inf) or negative rates: a negative custom rate
            # would LOWER session_cost and could slip a session under a USD ceiling
            # (same fail-under-count class as negative token counts). A malformed
            # spec is ignored entirely so the safe DEFAULT_RATES value stays in place.
            if not all(
                math.isfinite(r) and r >= 0
                for r in (candidate.input, candidate.output,
                          candidate.cache_write, candidate.cache_read)
            ):
                continue
            rates[family] = candidate
        return cls(rates=rates)
