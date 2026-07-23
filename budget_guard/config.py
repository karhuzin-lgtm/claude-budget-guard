"""Load limits and pricing from the environment (and an optional JSON file).

Configuration is *env-first* and *zero-config by default*: with nothing set the
hook is a pure no-op that always allows, so installing it can never
surprise-block a session. A limit only exists once you opt in by setting the
corresponding variable.

Resolution order (later wins):
    1. built-in defaults (no limits; example pricing)
    2. optional JSON file at ``CLAUDE_BUDGET_CONFIG``
    3. environment variables

Environment variables:
    CLAUDE_BUDGET_MAX_TOKENS  hard token ceiling (block at/above)      [unset]
    CLAUDE_BUDGET_MAX_USD     hard USD ceiling (block at/above)        [unset]
    CLAUDE_BUDGET_WARN_PCT    warn at this %% of a limit               [80]
    CLAUDE_BUDGET_LOOP_LIMIT  block when a tool call repeats N times   [unset=off]
    CLAUDE_BUDGET_CONFIG      path to a JSON overrides file            [unset]
    CLAUDE_BUDGET_OUTPUT      "json" to emit decision as stdout JSON   [stderr+exit2]

``CLAUDE_BUDGET_OUTPUT=json`` emits the official PreToolUse hook contract on
stdout and exits 0::

    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "<reason>"}}

(The deprecated top-level ``{"decision":"block"}`` form is NOT emitted — it may
be ignored for PreToolUse and let the blocked call through.) The default
``exit`` mode uses exit 2 + a stderr reason, which is the most robust path.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .pricing import PricingTable


@dataclass
class Config:
    """Resolved budget configuration."""

    max_tokens: Optional[int] = None
    max_usd: Optional[float] = None
    warn_pct: float = 80.0
    loop_limit: Optional[int] = None
    output_mode: str = "exit"  # "exit" (exit-2 + stderr) or "json" (stdout)
    pricing: PricingTable = field(default_factory=PricingTable)

    @property
    def has_any_limit(self) -> bool:
        """True if at least one guardrail is active. Otherwise the hook no-ops."""
        return (
            self.max_tokens is not None
            or self.max_usd is not None
            or (self.loop_limit is not None and self.loop_limit > 0)
        )


def _to_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _to_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    # Reject nan/inf: an infinite limit is "active" per has_any_limit yet can
    # never be reached, so the guard would look armed but never block.
    if not math.isfinite(result):
        return None
    return result if result > 0 else None


# Cap the config file we read. A config JSON is tiny; bound it so a FIFO/device
# or a huge file at CLAUDE_BUDGET_CONFIG can neither hang a synchronous read nor
# exhaust memory. Over the cap (or any error) we ignore the file -> defaults.
MAX_CONFIG_BYTES = 1024 * 1024  # 1 MiB, far above any real config


def _load_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    fd = None
    try:
        # Open non-blocking so a reader-less FIFO can't hang the open, then require
        # a REGULAR file (a FIFO/device/socket/dir is ignored -> defaults) before
        # reading at most the cap + 1 byte.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return {}
        raw = os.read(fd, MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            return {}  # oversized config -> ignore, fall back to defaults
        data = json.loads(raw.decode("utf-8", "replace"))
    except (OSError, ValueError, RecursionError):
        return {}
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    return data if isinstance(data, dict) else {}


def load_config(
    env: Optional[Mapping[str, str]] = None,
    file_data: Optional[dict] = None,
) -> Config:
    """Build a :class:`Config` from ``env`` (defaults to ``os.environ``).

    Never raises: malformed values fall back to their defaults so a typo in a
    variable cannot break the hook. ``file_data`` can be injected in tests;
    normally it is read from ``CLAUDE_BUDGET_CONFIG``.
    """
    env = os.environ if env is None else env

    if file_data is None:
        file_data = _load_file(env.get("CLAUDE_BUDGET_CONFIG"))

    # File provides the base layer.
    max_tokens = _to_int(file_data.get("max_tokens"))
    max_usd = _to_float(file_data.get("max_usd"))
    warn_pct = _to_float(file_data.get("warn_pct"))
    loop_limit = _to_int(file_data.get("loop_limit"))

    # Environment overrides the file.
    env_tokens = _to_int(env.get("CLAUDE_BUDGET_MAX_TOKENS"))
    if env_tokens is not None:
        max_tokens = env_tokens
    env_usd = _to_float(env.get("CLAUDE_BUDGET_MAX_USD"))
    if env_usd is not None:
        max_usd = env_usd
    env_warn = _to_float(env.get("CLAUDE_BUDGET_WARN_PCT"))
    if env_warn is not None:
        warn_pct = env_warn
    env_loop = _to_int(env.get("CLAUDE_BUDGET_LOOP_LIMIT"))
    if env_loop is not None:
        loop_limit = env_loop

    if warn_pct is None:
        warn_pct = 80.0
    # Clamp warn percentage to a sane 0-100 range.
    warn_pct = max(0.0, min(100.0, warn_pct))

    output_mode = "exit"
    if str(env.get("CLAUDE_BUDGET_OUTPUT", "")).strip().lower() == "json":
        output_mode = "json"

    pricing = PricingTable.from_mapping(file_data.get("pricing"))

    return Config(
        max_tokens=max_tokens,
        max_usd=max_usd,
        warn_pct=warn_pct,
        loop_limit=loop_limit,
        output_mode=output_mode,
        pricing=pricing,
    )
