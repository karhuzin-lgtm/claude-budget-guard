"""``python -m budget_guard`` -> run the PreToolUse hook.

Kept as a thin shim so the module entry and the console script share one code
path (``budget_guard.hook.main``).
"""

from __future__ import annotations

from .hook import main

if __name__ == "__main__":
    raise SystemExit(main())
