"""Turn measured usage into an allow / warn / block decision.

This is the policy core. It is intentionally pure — no I/O, no environment
access — so it is trivially testable and the hook can wrap it in a fail-open
try/except. Priority of outcomes:

    block  (over a hard ceiling, or a confirmed loop)  >  warn  >  allow

Only ``block`` maps to a non-zero exit for Claude Code; everything else lets the
tool call through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .loops import LoopInfo

ALLOW = "allow"
WARN = "warn"
BLOCK = "block"


@dataclass
class Decision:
    """The outcome for one hook invocation."""

    action: str = ALLOW
    reasons: List[str] = field(default_factory=list)
    tokens: int = 0
    usd: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    @property
    def message(self) -> str:
        return " ".join(self.reasons)


def _fmt_tokens(n: int) -> str:
    try:
        return f"{n:,}"
    except (ValueError, OverflowError):
        # Python caps int<->str conversion (~4300 digits); an absurdly large
        # count would raise here and, uncaught, fall into the hook's fail-open
        # path — allowing a CONFIRMED breach. Report an order-of-magnitude via
        # bit_length (no full string conversion) so the block still renders.
        approx = int(n.bit_length() * 0.30103) + 1 if n else 1
        return f"~10^{approx}"


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def decide(
    tokens: int,
    usd: float,
    config: Config,
    loop: Optional[LoopInfo] = None,
    loop_trustworthy: bool = True,
) -> Decision:
    """Evaluate thresholds and return a :class:`Decision`.

    ``tokens`` is the deduplicated session total; ``usd`` its estimated cost.
    ``loop`` is the result of :func:`budget_guard.loops.detect_loop`, or ``None``.

    ``loop_trustworthy`` gates whether a detected ``loop`` may BLOCK. Token/USD
    breaches are MONOTONIC — spend already seen stays confirmed even under a
    truncated/partial or content-dropping read — so they always block when over
    ceiling. A trailing retry-loop is NON-monotonic: an unread or skipped record
    could contain a different call that breaks the run, so a loop found on an
    incomplete/untrustworthy read is UNPROVEN and may only warn, never block
    (fail-open forbids blocking an unproven violation). The caller passes
    ``loop_trustworthy=False`` when the stream was ``partial`` or the trailing run
    was marked untrustworthy. The reasons are kept cleanly separated here so a
    partial read can still block on token/USD while never blocking on a loop.
    """
    decision = Decision(tokens=tokens, usd=usd)

    block_reasons: List[str] = []
    warn_reasons: List[str] = []

    # --- Hard token ceiling ------------------------------------------------
    if config.max_tokens is not None:
        if tokens >= config.max_tokens:
            block_reasons.append(
                "Token budget exceeded: session used "
                f"{_fmt_tokens(tokens)} tokens, limit is "
                f"{_fmt_tokens(config.max_tokens)}."
            )
        elif tokens >= config.max_tokens * config.warn_pct / 100.0:
            pct = tokens / config.max_tokens * 100.0
            warn_reasons.append(
                f"Token usage at {pct:.0f}% of budget "
                f"({_fmt_tokens(tokens)}/{_fmt_tokens(config.max_tokens)})."
            )

    # --- Hard USD ceiling --------------------------------------------------
    if config.max_usd is not None:
        if usd >= config.max_usd:
            block_reasons.append(
                "USD budget exceeded: estimated session cost "
                f"{_fmt_usd(usd)}, limit is {_fmt_usd(config.max_usd)} "
                "(estimate — verify pricing)."
            )
        elif usd >= config.max_usd * config.warn_pct / 100.0:
            pct = usd / config.max_usd * 100.0
            warn_reasons.append(
                f"Estimated cost at {pct:.0f}% of budget "
                f"({_fmt_usd(usd)}/{_fmt_usd(config.max_usd)})."
            )

    # --- Retry-loop --------------------------------------------------------
    # A loop may BLOCK only on a trustworthy (complete, non-dropping) read; an
    # unproven loop from a partial/untrustworthy read may warn but never block.
    if loop is not None:
        if loop_trustworthy:
            block_reasons.append(
                f"Retry-loop detected: tool '{loop.name}' repeated "
                f"{loop.count} times in a row (limit {config.loop_limit})."
            )
        else:
            warn_reasons.append(
                f"Possible retry-loop: tool '{loop.name}' repeated "
                f"{loop.count} times in a row (limit {config.loop_limit}), but the "
                "transcript read was incomplete/untrustworthy — allowing (unproven)."
            )

    if block_reasons:
        decision.action = BLOCK
        decision.reasons = block_reasons
    elif warn_reasons:
        decision.action = WARN
        decision.reasons = warn_reasons
    else:
        decision.action = ALLOW

    return decision
