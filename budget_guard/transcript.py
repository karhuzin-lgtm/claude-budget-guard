"""Parse a Claude Code session transcript (JSONL) into token usage.

Claude Code appends one JSON object per line to the session ``transcript_path``.
Each line has a ``"type"`` of ``"user"``, ``"assistant"`` or ``"system"``. Only
assistant lines carry a ``message.usage`` object with the token counts.

The single most important correctness detail lives here: **dedup by
requestId**. Claude Code writes ONE assistant line PER CONTENT BLOCK, and every
line from the same model response shares the same ``requestId`` and repeats the
*same* ``usage`` object. Summing every line inflates the token total 2-3x, so we
count each ``usage`` exactly once per unique request key (``requestId``, falling
back to ``message.id``; if neither is present the line is skipped rather than
risk double counting).

Parsing is deliberately defensive: a malformed line, a missing field or an
unexpected type is skipped, never fatal. The hook that calls us is fail-open,
and this module honours the same contract by returning whatever it could read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Usage:
    """Aggregated token counts for one or more model responses.

    Cache-creation and cache-read tokens are tracked separately from plain
    input tokens because they are billed at different rates (see ``pricing``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """All input-side tokens: fresh input + cache writes + cache reads."""
        return (
            self.input_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Grand total of every token that moved through the model."""
        return self.total_input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens
            + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass
class SessionUsage:
    """Usage for a whole session, broken down per model.

    A session can mix models (e.g. an Opus main loop with Haiku sub-agents), and
    USD cost depends on the model, so we keep a per-model breakdown alongside the
    request count. ``total`` collapses it to a single ``Usage`` for token limits.
    """

    by_model: Dict[str, Usage] = field(default_factory=dict)
    requests: int = 0

    def add(self, model: str, usage: Usage) -> None:
        current = self.by_model.get(model)
        self.by_model[model] = usage if current is None else current + usage
        self.requests += 1

    @property
    def total(self) -> Usage:
        total = Usage()
        for usage in self.by_model.values():
            total = total + usage
        return total

    @property
    def total_tokens(self) -> int:
        return self.total.total_tokens


@dataclass
class ToolCall:
    """A single ``tool_use`` block emitted by the assistant."""

    name: str
    input: dict


def tool_signature(call: "ToolCall") -> str:
    """Canonical identity of a tool call: name + sorted-key JSON of its input.

    Lives here (not in ``loops``) so the single-pass streamer can compute the
    trailing-run identity without importing the loop module (which would create
    an import cycle — ``loops`` imports from this module).
    """
    try:
        payload = json.dumps(call.input, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str is broad
        payload = repr(call.input)
    return f"{call.name}\x00{payload}"


@dataclass
class TrailingRun:
    """Running state for the *trailing* identical-call run.

    A trailing-run loop check needs only the last call's signature and how many
    times in a row it has repeated at the tail — NOT the whole tool-call list.
    Folding calls one at a time keeps loop detection O(1) memory in a single
    streaming pass over the transcript (see :func:`stream_session`).

    Two INDEPENDENT concerns are tracked, and they must never be conflated:

    * The RUN itself (``last``/``last_signature``/``count``) — how many identical
      calls end the observed stream. A mid-stream gap (a skipped record) BREAKS
      the run via :meth:`reset`, because the skipped record could have been a
      DIFFERENT call: the calls on either side are not provably consecutive, so a
      false "N in a row" must not fuse across the gap.

    * ``tail_suffix_unknown`` — whether the *unread suffix* of the stream is
      genuinely unknown. This is the ONLY thing that forbids a loop from BLOCKING
      (a loop is NON-monotonic, unlike a monotonic token/USD breach: unread
      content could contain a call that breaks the trailing run, so a loop over an
      unknown suffix is UNPROVEN and may only warn). It is set ONLY when there is
      really content we could not see AFTER what we read:
        (a) the stream was TRUNCATED before EOF — the line-count cap or the total
            byte budget tripped, or an overlong-line drain hit the byte budget;
        (b) a mid-read ``OSError``; or
        (c) the transcript was missing / non-regular / could not be opened.
      A mid-stream malformed/non-object/overlong-but-resynced line does NOT set
      it: that gap breaks the current run (:meth:`reset`) but a FRESH run forming
      entirely after the gap and fully observed to EOF is trustworthy and may
      block. Usage/dedup/model bounds (the stream's ``partial`` flag) also do NOT
      set it — they never drop a ``tool_use`` block, so the tail stays fully
      observed.

    :func:`budget_guard.hook.evaluate` consults ``tail_suffix_unknown`` (NOT
    ``partial``) before allowing a loop to block.
    """

    last: Optional[ToolCall] = None
    last_signature: Optional[str] = None
    count: int = 0
    tail_suffix_unknown: bool = False

    def add(self, call: ToolCall) -> None:
        sig = tool_signature(call)
        if self.last_signature is not None and sig == self.last_signature:
            self.count += 1
        else:
            self.last = call
            self.last_signature = sig
            self.count = 1

    def reset(self) -> None:
        """Break the trailing run across a mid-stream gap of unknown content.

        A skipped record could have been a DIFFERENT tool call, so the calls
        before and after it are not consecutive: drop the accumulated run so a
        false "N in a row" cannot form across the gap.

        This is a pure RUN discontinuity — it does NOT set
        ``tail_suffix_unknown``. A mid-stream gap does not make the *unread
        suffix* unknown: everything after it is still read to EOF, so a fresh run
        forming entirely after the gap is trustworthy and may block. Only a real
        truncation / read failure / unavailable transcript marks the tail unknown
        (see :func:`stream_session` and :func:`_iter_bounded_records`).
        """
        self.last = None
        self.last_signature = None
        self.count = 0


def _usage_from_dict(raw: object) -> Optional[Usage]:
    """Build a ``Usage`` from a raw ``message.usage`` dict, defensively.

    Returns ``None`` (skip the whole record) when any field is a non-finite
    float (nan/inf) — such a record is corrupt and cannot be trusted, so we
    decline to contribute anything rather than silently counting the good-looking
    fields around a poisoned one. Negative fields clamp to 0.

    A finite non-negative integer count is kept EXACTLY, however large. Python
    ints are arbitrary-precision, so the running token sum is always exact and
    can never overflow — a huge but real ``input_tokens`` (e.g. ``10**15``) must
    therefore still register against a token ceiling. An earlier version clamped
    every field DOWN to a fixed ``10**12`` bound to keep it finite for the float
    multiply in pricing; that clamp is REMOVED because it silently UNDER-COUNTED
    a genuine breach (``10**15`` tokens under a ``10**14`` limit clamped to
    ``10**12`` and wrongly ALLOWED — a fail-under-count). The USD path needs no
    clamp either: ``ModelPrice.cost`` already turns an un-priceable
    ``OverflowError`` into ``+inf`` ("over any finite USD ceiling"), so a huge
    finite count reads as over-budget on BOTH the token and USD axes, never as a
    swallowed error.
    """
    if not isinstance(raw, dict):
        return None

    fields = (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_creation_tokens", "cache_creation_input_tokens"),
        ("cache_read_tokens", "cache_read_input_tokens"),
    )
    values: Dict[str, int] = {}
    for out_key, in_key in fields:
        # A MISSING field is fine -> 0 (not every usage dict carries cache fields).
        if in_key not in raw:
            values[out_key] = 0
            continue
        value = raw[in_key]
        # STRICT typing: a token count must be a genuine non-negative int. Anything
        # else present (a numeric STRING like "999", a FLOAT, or a BOOL — note bool
        # is a subclass of int, so it is rejected explicitly) means the record is
        # malformed/untrustworthy. We DECLINE the whole record (return None) rather
        # than coerce a wrong-typed field into a "confirmed" count that could
        # falsely block, or into 0 that could hide real spend. Negatives clamp to 0
        # (a negative must never lower the total below a real breach). A valid
        # non-negative int is kept EXACTLY (Python big-int sum stays exact), so a
        # huge real count still trips the ceiling instead of being clamped under it.
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        values[out_key] = value if value >= 0 else 0

    return Usage(**values)


def _request_key(record: dict, message: dict) -> Optional[str]:
    """Dedup key for a response: requestId, else message.id, else None."""
    key = record.get("requestId") or record.get("request_id")
    if isinstance(key, str) and key:
        return key
    mid = message.get("id")
    if isinstance(mid, str) and mid:
        return mid
    return None


def iter_records(lines):
    """Yield parsed JSON objects from an iterable of text lines.

    Malformed or non-object lines are skipped silently. Accepts any iterable so
    callers can feed a file handle, a list, or a fixture string split by line.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            yield obj


def parse_records(records) -> SessionUsage:
    """Aggregate token usage from an iterable of transcript records.

    Applies the requestId dedup rule so each model response is counted once.
    """
    session = SessionUsage()
    seen = set()

    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        usage = _usage_from_dict(message.get("usage"))
        if usage is None:
            continue

        key = _request_key(record, message)
        if key is None:
            # No stable identity -> cannot dedup safely -> skip to avoid inflation.
            continue
        if key in seen:
            continue
        seen.add(key)

        model = message.get("model")
        if not isinstance(model, str) or not model:
            model = "unknown"
        session.add(model, usage)

    return session


def iter_tool_calls(records) -> List[ToolCall]:
    """Extract assistant ``tool_use`` blocks in chronological order.

    Used by loop detection. ``message.content`` is a list of blocks; each block
    with ``type == "tool_use"`` carries a ``name`` and ``input``. Non-list
    content, string content and unknown block shapes are ignored.
    """
    calls: List[ToolCall] = []
    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name:
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {}
            calls.append(ToolCall(name=name, input=tool_input))
    return calls


def _read_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def parse_transcript(path: str) -> SessionUsage:
    """Read a transcript file and return its deduplicated ``SessionUsage``.

    Raises ``OSError`` if the file cannot be opened; the caller (the hook) turns
    that into a fail-open allow.
    """
    return parse_records(iter_records(_read_lines(path)))


def load_tool_calls(path: str) -> List[ToolCall]:
    """Read a transcript file and return its ``tool_use`` blocks in order."""
    return iter_tool_calls(iter_records(_read_lines(path)))


# --- Bounded single-pass streaming -----------------------------------------
#
# The hot path (the hook) must NOT hold the whole transcript in memory: a huge
# or maliciously bloated JSONL session could OOM/timeout and get the process
# killed *before* the fail-open ``except`` runs — disabling the guard on exactly
# the long, expensive sessions it exists to protect.
#
# These bounds are chosen so normal sessions are entirely unaffected (real
# transcript lines are single-digit KB; sessions are thousands of lines). If a
# bound is hit the parse is INCOMPLETE: we stop/skip, stay fail-open (never
# crash), and flag the result ``partial`` so the caller never mistakes a
# truncated prefix for a complete session total. Chosen values:
#   * 4 MiB per physical line — orders of magnitude above any real line.
#   * 500,000 logical lines — a huge session is tens of thousands of lines.
#   * 256 MiB TOTAL raw bytes read (binary) — a HARD cap on the whole read so the
#     entire pass can never touch more than this, regardless of line count or a
#     single monstrous line drained in chunks. This is the master I/O bound.
#   * 500,000 distinct dedup keys and 10,000 distinct model names — COUNT caps.
#   * 64 MiB overall parser-state byte budget — a BYTE cap on ``seen`` +
#     ``by_model`` so total memory is bounded regardless of key sizes.
#
# The per-line + per-line-count caps alone were NOT enough for I/O: a single
# physical line drained as up to 500,000 chunks of 4 MiB, or 500,000 near-limit
# lines, still allowed ~2 TiB of reads and could hang/timeout the hook. The TOTAL
# byte budget (counted on RAW bytes in binary mode) is the real ceiling: once it
# trips we STOP reading and flag ``partial`` — we never keep draining. An overlong
# line is likewise abandoned immediately (drained only far enough to resynchronise
# to the next newline, itself bounded by the total budget), never up to the chunk
# cap.
#
# The count caps alone are NOT enough for MEMORY either: ``seen`` used to store
# raw ``requestId`` strings and ``by_model`` raw model names, both of arbitrary
# length, so a crafted transcript (each line a fresh multi-MiB key, all under the
# count cap) could accumulate gigabytes and OOM the process *before* the fail-open
# ``except`` ran. We now (a) store a FIXED-SIZE 16-byte digest of each dedup key
# instead of the raw string, (b) bound each model-name key (overlong names fold
# into a single fallback-priced sentinel bucket), and (c) enforce an overall
# byte budget. Hitting ANY of these flags ``partial`` (never a silent truncate).
DEFAULT_MAX_LINE_BYTES = 256 * 1024  # 256 KiB per line. A real transcript
# line is far smaller; capping it low bounds the memory json.loads can build
# from one line to a few MB worst case (no custom parser needed). Hardening
# against a maliciously CRAFTED transcript that forces OOM is out of the
# threat model: the transcript is written by Claude Code itself, not an
# untrusted external party. MemoryError is NOT caught below (it is not a
# recoverable syntax error) — it propagates and the hook fails open.
DEFAULT_MAX_LINES = 500_000
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # 256 MiB total raw bytes read
DEFAULT_MAX_DEDUP_KEYS = 500_000
DEFAULT_MAX_MODELS = 10_000
DEFAULT_MAX_STATE_BYTES = 64 * 1024 * 1024  # 64 MiB of parser-state memory

# Fixed digest size for a dedup key. blake2b-128 makes an accidental collision
# (two distinct requestIds -> same digest -> one wrongly dropped) negligible for
# dedup purposes, while bounding each ``seen`` entry to 16 bytes regardless of
# how long the raw requestId was.
_DEDUP_DIGEST_SIZE = 16

# Approximate per-entry container overhead (bytes) folded into the byte budget
# on top of the stored key's own size. A rough, deliberately conservative figure
# — the budget only needs to be an upper bound that trips before real memory is
# exhausted, not an exact sizeof.
_STATE_ENTRY_OVERHEAD = 64
_DEDUP_ENTRY_BYTES = _DEDUP_DIGEST_SIZE + _STATE_ENTRY_OVERHEAD

# Longest model name we keep verbatim as a ``by_model`` key. Real model names are
# ~15 chars; anything beyond this is hostile padding and folds into the sentinel.
MAX_MODEL_NAME_CHARS = 256

# Sentinel model name used when the distinct-model cap, the model-name-length cap
# or the byte budget is hit. Its tokens are still summed (never dropped); pricing
# falls back to the most-expensive rate.
_OVERFLOW_MODEL = "__overflow__"


def _dedup_digest(key: str) -> bytes:
    """Fixed-size digest of a dedup key (bounds ``seen`` memory; see above)."""
    return hashlib.blake2b(
        key.encode("utf-8", "replace"), digest_size=_DEDUP_DIGEST_SIZE
    ).digest()


class _Bounds:
    """Mutable state threaded through the streaming pass.

    ``partial`` flips to True the moment ANY bound trips (line bytes, line count,
    total bytes read, dedup-key count, model count, model-name length, or the
    overall parser-state byte budget), marking the parse as incomplete so the
    hook can warn instead of silently trusting a truncated total. ``state_bytes``
    is the running estimate of memory held by ``seen`` + ``by_model`` keys,
    checked against the state byte budget. ``total_bytes`` is the running count of
    RAW bytes read from the file (binary), checked against the total I/O budget.
    """

    __slots__ = ("partial", "state_bytes", "total_bytes")

    def __init__(self) -> None:
        self.partial = False
        self.state_bytes = 0
        self.total_bytes = 0


def _drain_overlong(
    fh,
    max_line_bytes: int,
    max_total_bytes: int,
    bounds: "_Bounds",
    run: "TrailingRun",
) -> None:
    """Resynchronise past the remainder of an overlong physical line.

    Reads forward in ``max_line_bytes``-sized chunks (never buffering the whole
    line) until the next newline / EOF, charging every raw byte to the TOTAL byte
    budget. If the budget trips mid-drain we STOP immediately — a monstrous line
    can therefore never cause more than ``max_total_bytes`` of reads, so there is
    no ~TiB drain and no hang. The overlong line itself was already flagged
    ``partial`` and reset the run by the caller.

    Two distinct exits, with DIFFERENT trust consequences:
      * Budget tripped mid-drain -> the stream is TRUNCATED: there is real unread
        content after this point, so mark ``run.tail_suffix_unknown`` (a loop can
        no longer be proven). Also ``partial`` (usage incomplete).
      * Reached the next newline / EOF -> the overlong line was fully skipped and
        we RESYNCED; the suffix after it is still read normally, so the tail stays
        trustworthy (``reset`` on the caller already broke the run across the gap).
    """
    while True:
        if bounds.total_bytes >= max_total_bytes:
            bounds.partial = True
            run.tail_suffix_unknown = True  # truncated: real unread content remains
            return
        more = fh.readline(max_line_bytes + 1)
        if more == b"":
            return
        bounds.total_bytes += len(more)
        if more.endswith(b"\n"):
            return


def _iter_bounded_records(
    fh,
    max_line_bytes: int,
    max_lines: Optional[int],
    max_total_bytes: int,
    bounds: "_Bounds",
    run: "TrailingRun",
):
    """Yield parsed dict records from a BINARY ``fh`` with hard I/O/size bounds.

    Reads raw bytes with ``readline(limit)`` and decodes each line with
    ``errors="replace"``, so a single pathologically long physical line (no
    newlines, gigabytes) is read in bounded chunks and skipped — never buffered
    whole. Three I/O bounds guard the pass:

      * ``max_total_bytes`` — the MASTER cap on total RAW bytes read. Once the
        running ``bounds.total_bytes`` reaches it we STOP the whole pass and flag
        ``partial`` — we never keep draining (this is what caps the old ~2 TiB
        worst case).
      * ``max_line_bytes`` — an overlong physical line is abandoned at once: flag
        ``partial``, reset the run, then :func:`_drain_overlong` reads only far
        enough to resynchronise to the next newline (itself total-byte bounded).
      * ``max_lines`` — a logical-line-count cap; tripping it stops early
        (``partial``).

    Whenever a record is SKIPPED (overlong/dropped line, malformed JSON, or a
    non-object), the trailing tool-call run is RESET via ``run.reset`` — the
    skipped record could have been a different tool call, so the calls around the
    gap are not consecutive and must not be fused into a false loop. A mid-stream
    gap only breaks the run; it does NOT mark the tail unknown, because everything
    after it is still read to EOF (a fresh run forming after the gap stays
    trustworthy). Only a genuine TRUNCATION — the line-count cap, the total byte
    budget, or an overlong-line drain that hits the byte budget — leaves real
    unread content after what we saw, and only those set ``tail_suffix_unknown``.
    (Blank/whitespace-only lines carry no content — trustworthy emptiness — so
    they do NOT reset the run.)
    """
    lines_seen = 0
    while True:
        if max_lines is not None and lines_seen >= max_lines:
            # Stopped early: more lines remain unread -> usage incomplete AND the
            # tail suffix is genuinely unknown (a loop can no longer be proven).
            bounds.partial = True
            run.tail_suffix_unknown = True
            break
        if bounds.total_bytes >= max_total_bytes:
            # Total I/O budget exhausted: stop reading. Real unread content
            # remains -> incomplete usage AND unknown tail suffix.
            bounds.partial = True
            run.tail_suffix_unknown = True
            break
        segment = fh.readline(max_line_bytes + 1)
        if segment == b"":
            break
        bounds.total_bytes += len(segment)
        lines_seen += 1
        # A physical line longer than the cap comes back at the cap length with
        # no trailing newline. Abandon it immediately: flag partial, break the
        # run (mid-stream gap), and drain only enough to reach the next newline
        # (total-byte bounded) so a monstrous line can never be read past the
        # budget. The drain marks the tail unknown ONLY if it hits the budget
        # (true truncation); a clean resync to the next newline leaves the tail
        # trustworthy.
        if len(segment) > max_line_bytes and not segment.endswith(b"\n"):
            bounds.partial = True
            run.reset()
            _drain_overlong(fh, max_line_bytes, max_total_bytes, bounds, run)
            continue
        stripped = segment.strip()
        if not stripped:
            # Blank line: no content, nothing dropped -> trustworthy, no reset.
            continue
        try:
            obj = json.loads(stripped.decode("utf-8", "replace"))
        except (ValueError, RecursionError):
            # ANY parse failure (malformed JSON -> ValueError; deeply nested JSON
            # -> RecursionError; and any other parser exception) drops this record.
            # We catch broadly ON PURPOSE: a single crafted line must never escape
            # to the hook's outer handler and fail-open the whole guard (which would
            # let an already-confirmed breach through). Unknown content dropped ->
            # break the run so the calls around it are not fused into a false loop.
            # This is a mid-stream gap: it does NOT mark the tail unknown (content
            # after it is still read to EOF, so a fresh post-gap run may block).
            # (MemoryError is a BaseException and deliberately not caught — it is
            # not reliably recoverable mid-process.)
            run.reset()
            continue
        if isinstance(obj, dict):
            yield obj
        else:
            # Non-object JSON (list, number, ...): still a dropped record -> reset.
            run.reset()


def _fold_record(
    record: dict,
    session: SessionUsage,
    seen: set,
    run: TrailingRun,
    bounds: "_Bounds",
    max_dedup_keys: int = DEFAULT_MAX_DEDUP_KEYS,
    max_models: int = DEFAULT_MAX_MODELS,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> None:
    """Fold ONE assistant record into the running usage + trailing-run state.

    Usage aggregation applies the requestId dedup (identical to
    :func:`parse_records`, but keyed on a fixed-size digest of the request key so
    ``seen`` cannot grow with key length); tool-call folding counts every
    ``tool_use`` block (identical to :func:`iter_tool_calls`) so the streamed
    result matches the old two-pass computation exactly. Any memory bound
    (dedup-key count, model count, model-name length, overall byte budget) flips
    ``bounds.partial``.

    RECORD-level un-interpretability breaks the trailing run exactly like a
    line-level skip: an assistant record that COULD carry a tool call but cannot
    be reliably read — ``message`` not a dict, ``content`` not a list, or a
    ``tool_use`` block with a non-str/empty ``name`` or a non-dict ``input`` —
    calls ``run.reset()`` (a mid-stream gap that BREAKS the run but does not mark
    the tail unknown) rather than being silently ignored, because otherwise two
    identical calls on either side of an un-interpretable tool_use would fuse into
    a false "confirmed" consecutive run. An invalid
    ``tool_input`` is NOT coerced to ``{}`` (which could fabricate a matching
    signature) — it breaks the run instead. Records that legitimately carry NO
    tool_use block (a pure-text assistant line, ``content`` a list without a
    tool_use) do NOT reset — that interleaving is normal between real retries.
    Non-assistant records (user/system tool_result lines) are not tool carriers
    and never reset the run.
    """
    if record.get("type") != "assistant":
        return
    message = record.get("message")
    if not isinstance(message, dict):
        # An assistant record whose message is un-interpretable could have held a
        # tool_use: break the run so it cannot fuse a false consecutive loop.
        run.reset()
        return

    # (a) deduplicated usage
    usage = _usage_from_dict(message.get("usage"))
    if usage is not None:
        key = _request_key(record, message)
        if key is not None:
            digest = _dedup_digest(key)
            if digest not in seen:
                # Bound both the COUNT of dedup keys and the total BYTES of parser
                # state before admitting a new identity. Hitting either stops
                # counting new requests (rather than risk double-counting or
                # unbounded memory) and flags the result incomplete.
                if (
                    len(seen) >= max_dedup_keys
                    or bounds.state_bytes + _DEDUP_ENTRY_BYTES > max_state_bytes
                ):
                    bounds.partial = True
                else:
                    seen.add(digest)
                    bounds.state_bytes += _DEDUP_ENTRY_BYTES
                    model = message.get("model")
                    if not isinstance(model, str) or not model:
                        model = "unknown"
                    # Bound the model-name key length: an overlong (hostile) name
                    # folds into the sentinel bucket so ``by_model`` keys stay
                    # small (tokens still summed, priced at the conservative
                    # fallback).
                    if len(model) > MAX_MODEL_NAME_CHARS:
                        bounds.partial = True
                        model = _OVERFLOW_MODEL
                    if model not in session.by_model:
                        model_cost = len(model) + _STATE_ENTRY_OVERHEAD
                        # Cap distinct model count AND the byte budget: overflow
                        # folds into the sentinel bucket.
                        if (
                            len(session.by_model) >= max_models
                            or bounds.state_bytes + model_cost > max_state_bytes
                        ):
                            bounds.partial = True
                            model = _OVERFLOW_MODEL
                        # Charge bytes only for a genuinely new bucket (the
                        # sentinel may already exist).
                        if model not in session.by_model:
                            bounds.state_bytes += len(model) + _STATE_ENTRY_OVERHEAD
                    session.add(model, usage)

    # (b) trailing tool-call run (every tool_use block, in order)
    content = message.get("content")
    if not isinstance(content, list):
        # An assistant record whose content is not a list is un-interpretable as
        # a tool carrier: break the run (a real assistant line always carries a
        # list of content blocks). Pure-text lines have content == a LIST and are
        # handled by the loop below (no tool_use -> no reset).
        run.reset()
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        tool_input = block.get("input")
        # A tool_use block that cannot be interpreted (bad name or non-dict
        # input) BREAKS the run — it is un-interpretable content between calls.
        # Never coerce a bad input to {}: that could fabricate a signature that
        # falsely matches the surrounding calls into a confirmed loop.
        if not isinstance(name, str) or not name or not isinstance(tool_input, dict):
            run.reset()
            continue
        run.add(ToolCall(name=name, input=tool_input))


def stream_session(
    path: str,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    max_lines: Optional[int] = DEFAULT_MAX_LINES,
    max_dedup_keys: int = DEFAULT_MAX_DEDUP_KEYS,
    max_models: int = DEFAULT_MAX_MODELS,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
):
    """Single bounded pass over a transcript file.

    Returns ``(SessionUsage, TrailingRun, partial)`` — the deduplicated usage,
    the trailing identical-call run, and a ``partial`` flag that is True when any
    bound (line bytes, line count, TOTAL bytes read, dedup-key/model count,
    model-name length or the parser-state byte budget) tripped and the parse is
    therefore INCOMPLETE (usage may under-count). The returned ``TrailingRun``
    additionally carries a ``tail_suffix_unknown`` flag — set ONLY when the unread
    SUFFIX is genuinely unknown: a truncation (line-count / total-byte / overlong-
    drain budget), a mid-read ``OSError``, or an unavailable transcript. A
    mid-stream skipped record breaks the run but does NOT set it, and usage/dedup/
    model bounds (``partial``) do NOT set it either. The caller must treat
    ``tail_suffix_unknown`` — NOT ``partial`` — as the signal that a loop cannot be
    CONFIRMED (loops are non-monotonic), while a token/USD breach on the read
    portion stays confirmed under a partial read. The single pass runs from the
    start of the file so the token sum and trailing run cover as much as possible;
    ``partial`` warns the caller never to treat a truncated prefix as a complete
    total.

    The file is opened in BINARY mode and every line decoded with
    ``errors="replace"`` so byte budgets are enforced on RAW bytes. Before any
    read the opened descriptor is ``fstat``-ed and required to be a REGULAR file:
    a FIFO/device/socket/dir is fail-open ALLOWED (empty usage, ``partial=False``)
    without reading, so a synchronous hook can never hang on a ``readline`` of a
    pipe that may never return. The open uses ``O_NONBLOCK`` so even opening a
    reader-less FIFO returns immediately instead of blocking (``O_NONBLOCK`` has
    no effect on regular-file reads).

    Raises ``OSError`` if the file cannot be opened; the caller (the hook) turns
    that into a fail-open allow.
    """
    session = SessionUsage()
    seen: set = set()
    run = TrailingRun()
    bounds = _Bounds()

    # Open non-blocking so a reader-less FIFO does not hang the open itself, then
    # verify it is a regular file before reading. os.open may raise OSError, which
    # the caller turns into a fail-open allow.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            # Non-regular (FIFO/device/socket/dir): never risk a blocking read.
            # Nothing was read, so the ENTIRE suffix is unknown, not a proven
            # empty run — mark ``tail_suffix_unknown`` so a loop can never BLOCK
            # off it (case (c)). Token/USD stay 0 here (an unread transcript cannot
            # breach a ceiling), so only the loop path needed hardening.
            run.tail_suffix_unknown = True
            os.close(fd)
            return session, run, False
        fh = os.fdopen(fd, "rb")
    except OSError:
        # Could not stat/fdopen: same as a failed read — the suffix is unknown,
        # so mark ``tail_suffix_unknown`` (case (c)) before the fail-open return.
        run.tail_suffix_unknown = True
        try:
            os.close(fd)
        except OSError:
            pass
        return session, run, False

    with fh:
        try:
            for record in _iter_bounded_records(
                fh, max_line_bytes, max_lines, max_total_bytes, bounds, run
            ):
                _fold_record(
                    record, session, seen, run, bounds,
                    max_dedup_keys, max_models, max_state_bytes,
                )
        except OSError:
            # A read error PART-WAY through (e.g. a network/FUSE-backed transcript
            # that streamed some lines then failed) must NOT discard the usage
            # already accumulated: prefix spend is monotonic, so a breach found so
            # far still stands. Keep `session`/`run`, flag the read INCOMPLETE, and
            # mark the tail unknown (case (b): its suffix could not be read). A
            # pre-read open failure is handled above and returns an empty result;
            # only a mid-read failure reaches here.
            bounds.partial = True
            run.tail_suffix_unknown = True
    return session, run, bounds.partial
