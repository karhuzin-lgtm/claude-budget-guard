"""claude-budget-guard — a preventive spend guardrail for Claude Code.

Most cost tooling for Claude Code is *observational*: dashboards that tell you
what a session already spent. By then the money is gone. ``budget_guard`` runs
as a ``PreToolUse`` hook and intervenes *before* each tool call — it reads the
live session transcript, sums the (deduplicated) token usage, and blocks the
call when the session has crossed a hard token/USD budget or is stuck in a
retry-loop.

Design posture: **fail-open**. A monitoring guard must never break a working
session because of its own bug, so any error (unreadable transcript, malformed
config, parse failure) results in *allow*. The only paths that block are a
*confirmed* over-budget condition or a *confirmed* loop.

Public API:
    - ``Usage`` / ``SessionUsage``  (token accounting, ``transcript`` module)
    - ``parse_transcript``          (JSONL -> ``SessionUsage``)
    - ``PricingTable``              (tokens -> USD, ``pricing`` module)
    - ``Config``                    (limits + pricing from env/JSON)
    - ``decide`` / ``Decision``     (threshold logic, ``budget`` module)
    - ``detect_loop``               (retry-loop detection, ``loops`` module)
    - ``main``                      (hook entry point, ``hook`` module)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Usage",
    "SessionUsage",
    "parse_transcript",
    "PricingTable",
    "Config",
    "decide",
    "Decision",
    "detect_loop",
    "main",
]


def __getattr__(name: str):
    # Lazy re-exports keep import cost minimal for the hot hook path.
    if name in ("Usage", "SessionUsage", "parse_transcript"):
        from . import transcript

        return getattr(transcript, name)
    if name == "PricingTable":
        from .pricing import PricingTable

        return PricingTable
    if name == "Config":
        from .config import Config

        return Config
    if name in ("decide", "Decision"):
        from . import budget

        return getattr(budget, name)
    if name == "detect_loop":
        from .loops import detect_loop

        return detect_loop
    if name == "main":
        from .hook import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
