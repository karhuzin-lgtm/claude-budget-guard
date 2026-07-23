# claude-budget-guard

**A preventive spend guardrail for Claude Code — the seatbelt for AI coding agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![deps: stdlib only](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)
[![tests: unittest](https://img.shields.io/badge/tests-156%20passing-brightgreen.svg)](#running-the-tests)

Every other cost tool for Claude Code *observes* — it shows you the bill after
the money is gone. `claude-budget-guard` *intervenes*. It runs as a `PreToolUse`
hook and stops a session **before** it blows a token or dollar budget, or before
it burns another hour stuck in a retry-loop.

---

## The problem

An agent can silently drain an entire weekly quota in a single session, or loop
for hours as a black box — re-running the same failing command, retrying the
same request — with no brake anywhere in the loop. Dashboards tell you what it
cost *after* it happened. That is a smoke detector wired to sound only once the
house has burned down.

`claude-budget-guard` is the brake. It sits in front of every tool call, reads
the live session transcript, and refuses to let a call through once the session
has crossed a hard budget or is demonstrably looping.

## How it works

Claude Code runs the hook before each tool call and hands it the session's
transcript path on stdin. The hook sums the session's **deduplicated** token
usage and decides:

```
                 ┌─────────────────────────────────────────────┐
   tool call ──▶ │  PreToolUse hook: python3 -m budget_guard    │
                 │                                              │
                 │  read transcript ─▶ dedup by requestId ─▶    │
                 │  sum tokens (+ optional USD) ─▶ check loop    │
                 └───────────────┬──────────────────────────────┘
                                 │
          under budget ──────────┼──────────▶  allow      (exit 0)
          crossed warn %  ───────┼──────────▶  warn+allow (exit 0, stderr note)
          over hard budget ──────┼──────────▶  BLOCK      (exit 2, reason→agent)
          retry-loop detected ───┴──────────▶  BLOCK      (exit 2, reason→agent)
```

A blocked call exits `2` and writes the reason to stderr; Claude Code shows that
reason to the agent, so the model learns *why* it was stopped.

**Fail-open by design.** If the hook can't read the transcript, can't parse a
line, or hits any error at all, it allows the call. A guardrail must never be
the reason a healthy session dies. The *only* paths to a block are a confirmed
over-budget condition or a confirmed loop.

### The dedup detail

Claude Code writes one transcript line **per content block**, and every line
from the same model response repeats the *same* `usage` object under the same
`requestId`. Summing every line inflates your token count 2–3×. The guard counts
each `usage` exactly once per unique `requestId` (falling back to `message.id`).
This is the single most important correctness detail in the tool, and it is
covered by explicit tests.

## Install

Requires Python 3.8+. No dependencies.

```bash
pip install claude-budget-guard
# or, from a clone:
pip install .
```

You can also run it straight from a checkout with no install at all — it's
stdlib-only, so `python3 -m budget_guard` works from the repo root.

## Quickstart

Wire it as a `PreToolUse` hook in your project's `.claude/settings.json`, then
set a budget with environment variables:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "python3 -m budget_guard" }
        ]
      }
    ]
  }
}
```

Set at least one limit (otherwise the hook is a deliberate no-op):

```bash
# Block the session once it has used 2,000,000 tokens; warn at 80%.
export CLAUDE_BUDGET_MAX_TOKENS=2000000

# Optionally add a dollar ceiling and loop protection.
export CLAUDE_BUDGET_MAX_USD=25
export CLAUDE_BUDGET_LOOP_LIMIT=10
```

That's it. Under budget, you'll never notice it. Cross 80% and you get a
one-line warning. Cross the ceiling, or repeat the same tool call ten times in a
row, and the next call is blocked with an explanation.

## Configuration

Everything is environment-first and zero-config by default. **With nothing set,
the hook always allows** — installing it can never surprise-block you.

| Variable | Meaning | Default |
| --- | --- | --- |
| `CLAUDE_BUDGET_MAX_TOKENS` | Hard token ceiling. Block at/above this total (deduped input + output + cache tokens). | unset → no token limit |
| `CLAUDE_BUDGET_MAX_USD` | Hard USD ceiling. Block at/above this estimated cost. | unset → no USD limit |
| `CLAUDE_BUDGET_WARN_PCT` | Warn (but still allow) at this percentage of a limit. | `80` |
| `CLAUDE_BUDGET_LOOP_LIMIT` | Block when the identical tool call repeats this many times in a row. | unset → loop detection **off** |
| `CLAUDE_BUDGET_CONFIG` | Path to a JSON file overriding limits and/or pricing. | unset |
| `CLAUDE_BUDGET_OUTPUT` | `json` to emit the official PreToolUse deny payload on stdout (exit 0) instead of exit-2 + stderr (see below). | exit-2 + stderr |

### JSON config file

`CLAUDE_BUDGET_CONFIG` points to a file like this. Environment variables
override anything set here.

```json
{
  "max_tokens": 2000000,
  "max_usd": 25,
  "warn_pct": 80,
  "loop_limit": 10,
  "pricing": {
    "opus":   { "input": 15,  "output": 75, "cache_write": 18.75, "cache_read": 1.5 },
    "sonnet": { "input": 3,   "output": 15, "cache_write": 3.75,  "cache_read": 0.3 },
    "haiku":  { "input": 0.8, "output": 4,  "cache_write": 1.0,   "cache_read": 0.08 }
  }
}
```

Pricing keys match as case-insensitive substrings of the model name (so `opus`
matches `claude-opus-4-8`). Rates are per **million** tokens.

### JSON output mode

With `CLAUDE_BUDGET_OUTPUT=json`, a block is emitted on **stdout** with exit `0`
using the official Claude Code PreToolUse hook contract:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "[budget-guard] Token budget exceeded: ..."
  }
}
```

(`permissionDecision` is one of `allow` / `deny` / `ask`; the deprecated
top-level `{"decision":"block"}` form is **not** emitted — for PreToolUse it may
be ignored, which would let the blocked call through.) The default `exit` mode —
exit `2` with the reason on stderr — is the most robust path and is recommended.

### Partial reads

The hook streams the transcript in a single bounded pass so a pathologically
large or bloated session can never OOM the guard. If a bound trips (a >4 MiB
physical line, more than 2,000,000 lines, or the dedup-key/model memory caps),
the parse is **incomplete**. A confirmed breach on the portion that *was* read
still blocks; if no breach is provable on the partial data the hook allows the
call but prints `WARNING: transcript too large to fully account (partial read)`
to stderr — it never silently treats a truncated prefix as a full total.

## What it detects

- **Token ceiling** — the primary, deterministic metric. Sums deduped input,
  output, cache-write and cache-read tokens across the whole session.
- **USD ceiling** — an optional cost estimate layered on top, priced per model.
- **Warn threshold** — a soft heads-up at a configurable percentage of any limit;
  never blocks.
- **Retry-loop** — the identical tool call (same name + same input) repeating N
  times in a row, the classic "agent stuck burning tokens" failure.

## A note on pricing

Tokens are the primary budget metric precisely because they are deterministic
and never go stale. USD is a **convenience estimate** and only as accurate as
the rates you give it. The built-in rates are **example figures for
illustration** — Anthropic pricing changes, so verify current pricing and
override the rates via `CLAUDE_BUDGET_CONFIG` before you rely on a dollar limit.
No number in this repo is presented as authoritative.

## Why it's different

Existing tools observe your spend. This one intervenes before it happens.

## Running the tests

Pure stdlib, no test runner required:

```bash
python -m unittest discover -s tests
```

`pytest` works too if you prefer it. Coverage includes the requestId dedup rule,
token summation, pricing math, threshold decisions (under / warn / over), loop
detection, and the end-to-end hook exit codes, plus every fail-open edge case
(empty transcript, missing file, garbage stdin, no limits configured).

## License

MIT © Aleksei. See [LICENSE](LICENSE).
