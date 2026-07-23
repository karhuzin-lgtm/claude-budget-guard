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
    return f"{n:,}"


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def decide(
    tokens: int,
    usd: float,
    config: Config,
    loop: Optional[LoopInfo] = None,
) -> Decision:
    """Evaluate thresholds and return a :class:`Decision`.

    ``tokens`` is the deduplicated session total; ``usd`` its estimated cost.
    ``loop`` is the result of :func:`budget_guard.loops.detect_loop`, or ``None``.
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
    if loop is not None:
        block_reasons.append(
            f"Retry-loop detected: tool '{loop.name}' repeated "
            f"{loop.count} times in a row (limit {config.loop_limit})."
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
