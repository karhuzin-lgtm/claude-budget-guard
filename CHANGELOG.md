# Changelog

All notable changes to **claude-budget-guard** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project aims to adhere to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-23

### Added
- Initial release: a preventive spend guardrail for Claude Code, delivered as a
  `PreToolUse` hook.
- Token ceiling, USD ceiling (configurable per-model pricing), warn threshold,
  and retry-loop detection.
- Deduplicated token accounting (by `requestId`) — counts each model response
  once instead of inflating 2–3× per content block.
- Single bounded streaming pass over the transcript: total-byte budget,
  overlong-line handling, and an `S_ISREG` guard so a FIFO/special file can never
  hang the hook.
- Zero-config no-op by default, absolute fail-open on error, and a hardened block
  path: a confirmed breach still exits `2` even if the output stream errors.
- JSON output mode using the official `hookSpecificOutput.permissionDecision`
  contract.
- 156-test stdlib `unittest` suite.

[0.1.0]: https://github.com/karhuzin-lgtm/claude-budget-guard/releases/tag/v0.1.0
