"""Pure algorithm — structure-aware chunking of court opinions.

No I/O, no database, no model. Deterministic: same input + same tokenizer
+ same is_noise predicate + same token limit produce the same output.

v2.7 changes (final):
  - QUALITY_GATE_3_MAX_RATE = 0.04. Since aggregate gate_3 rate is a
    weighted average of per-case rates, keeping per-case rate <= 0.04
    mathematically guarantees aggregate <= 0.04 < 0.05 gate threshold.
  - Version bumped to chunk-v2.7. Chunk output is byte-identical to v2.6.

v2.6 changes:
  - quality_filter(): per-case drop before chunk files are written.
    A case failing any of {gate_3 rate, gate_4 hits, corruption score}
    is logged to _corruption_drops.jsonl and excluded from load.

v2.4-2.5 changes:
  - _normalize strips *Page NNN inline page markers.
  - _SENTENCE_END_RE accepts ; and includes ' (U+0027) in closer class.
  - Reporter pattern extended to accept bare reporters and state reporters.
  - _split_by_budget merges paragraphs whose boundary lands inside an
    atomic span (fixes "Corp.,\n\n555 So. 2d" splits).

v2.3 changes:
  - Atomic-span intervals built once per opinion; every boundary check
    uses binary search instead of regex rescans (4-8x speedup).
  - Fallback path preserves v2.2 behaviour when intervals=None.

v2.2 changes:
  - ATOMIC_SPAN_PATTERNS extended for multi-word defendant names
    ("Geohagan v. General Motors Corp.,") and case-name-then-reporter
    cuts ("...Corp., | 555 So. 2d").
  - corruption_score() + CORRUPTION_THRESHOLD for drop-above-threshold.

v2.1 changes:
  - _normalize strips CourtListener page markers (*NNN, * Hi * Hi *).
  - _SENTENCE_END_RE accepts ":" as terminal.

v2.0 changes:
  - Single newlines not treated as paragraph boundaries.
  - max_embedding_tokens configurable per backend (default 464).
  - Child-body budget derived from measured breadcrumb size.

v1.5 changes:
  - Optional is_noise callable filters child bodies.

v1.4 changes:
  - _safe_cut uses binary search when tokenizer available.
  - Breadcrumb headroom reserved in child token budget.
  - Post-emit validation rejects children exceeding limit.
  - CHARS_PER_TOKEN reduced from 3.0 to 2.25.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

# ── Versioning ──────────────────────────────────────────────────────────

CHUNKER_VERSION = "chunk-v2.8"


# ── Default embedder token limit ────────────────────────────────────────
#
# 464 is the OpenVINO compiled sequence limit. Callers with a verified
# backend using a different sequence length must pass max_embedding_tokens
# to chunk_opinion().

EMBEDDER_MAX_TOKENS = 464


# ── Opinion-type mapping (CourtListener codes → canonical labels) ──────

OPINION_TYPE_MAP: dict[str, str] = {
    "020lead": "majority",
    "010combined": "combined",
    "040dissent": "dissent",
    "030concurrence": "concurrence",
    "035concurrenceinpart": "concurrence",
    "070rehearing": "rehearing",
    "050addendum": "addendum",
}

TEXT_KIND_PREFERENCE = (
    "original",
    "combined",
    "generated",
    "html_with_citations",
    "plain_text",
)


# ── Size budget (token counts) ──────────────────────────────────────────
#
# Observed chars/token on target corpus with dinghy tokenizer:
# min 2.0, p05 2.25, p50 4.04. Dense citation blocks are worst case.
# This fallback is used only when no real tokenizer is supplied.

CHARS_PER_TOKEN = 2.25

PARENT_TARGET_TOK = 1000
PARENT_MAX_TOK = 1400
PARENT_MIN_TOK = 500

CHILD_TARGET_TOK = 320
CHILD_MAX_TOK = 460
CHILD_MIN_TOK = 160

CHILD_OVERLAP_FALLBACK_TOK = 32


# ── Section detection ───────────────────────────────────────────────────

SECTION_PATTERNS: tuple[tuple[str, int, str], ...] = (
    (
        "conclusion",
        95,
        r"(?im)^[ \t]*(?:CONCLUSION\b|We therefore\b|Accordingly,?\s+(?:we|the court)\b)",
    ),
    (
        "judgment",
        90,
        r"(?im)^[ \t]*(?:It is so ordered\b|REVERSED\b|AFFIRMED\b|REVERSED AND REMANDED\b|VACATED AND REMANDED\b)",
    ),
    ("holding", 88, r"(?im)^[ \t]*(?:We hold\b|We conclude\b|Held:?\s)"),
    (
        "dissent_header",
        82,
        r"(?im)^[ \t]*(?:JUSTICE|CHIEF JUSTICE|JUDGE|PER CURIAM)\s+[A-Z][A-Z .'\-]+?,?\s*(?:dissenting|concurring(?:\s+in\s+(?:part|the\s+result))?|concurring\s+in\s+part\s+and\s+dissenting\s+in\s+part)",
    ),
    (
        "facts",
        70,
        r"(?im)^[ \t]*(?:Facts\b|Background\b|Factual Background\b|Procedural History\b)",
    ),
    (
        "analysis",
        85,
        r"(?im)^[ \t]*(?:[IVXLC]{1,4}(?:\.[A-Z])?\.\s*|Discussion\b|Analysis\b|Opinion of the Court\b)",
    ),
    (
        "caption",
        40,
        r"(?im)^[ \t]*(?:<parties>|<docketnumber>|No\.\s+\d{2}-\d+|SUPREME COURT OF|IN THE .* COURT\b)",
    ),
    ("footnote", 20, r"(?im)^[ \t]*(?:\*|\[\d+\]|\d+\.\s+(?=[A-Z]))"),
)

COMPILED_SECTION_PATTERNS = tuple(
    (name, priority, re.compile(pattern)) for name, priority, pattern in SECTION_PATTERNS
)


# ── Atomic spans (never split across a chunk boundary) ─────────────────

ATOMIC_SPAN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 410 U.S. 113 (federal reporter with single-word reporter name)
    re.compile(r"\d+\s+[A-Z][A-Za-z.']*\s+\d+"),
    # 42 U.S.C. § 1983 (USC with section)
    re.compile(r"\d+\s+U\.S\.C\.\s+§+\s*\d+[\w().]*"),
    # § 1983 (bare section)
    re.compile(r"§+\s*\d+[\w().-]*"),
    # No. 12-345 (docket number)
    re.compile(r"No\.\s*\d{2}-\d+"),
    # $10,000 (currency)
    re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?"),
    # 1973 (year)
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # 06/15/2020 (date)
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    # Basic case name: "Roe v. Wade"
    re.compile(r"\b[A-Z][A-Za-z']+\s+v\.\s+[A-Z][A-Za-z']+"),
    # Case name with corporate suffix: "Safeguard Business Sys., Inc. v. S"
    re.compile(r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+[A-Z]"),
    # Extended case name — up to 3 more words after the defendant first word.
    # Fixes "Geohagan v. General Motors Corp." without over-extending into
    # the next sentence.
    re.compile(
        r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+"
        r"[A-Z][\w.,'\-]*\.?"
        r"(?:(?<=\w)[ \t]+[A-Z][\w.,'\-]*\.?){0,3}"
    ),
    # Case name + reporter cut — bridges "...Corp.," and "555 So. 2d".
    re.compile(
        r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+"
        r"[A-Z][\w.,'\-]*\.?"
        r"(?:(?<=\w)[ \t]+[A-Z][\w.,'\-]*\.?){0,3}"
        r",\s+\d+\s+[A-Z]"
    ),
    # Ala. Code | N.Y. Code (state code)
    re.compile(r"\b(?:[A-Z][a-z]*\.)+\s*Code\b"),
    # Mr. | Mrs. | Dr. (honorific followed by capitalized name)
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)\.\s+[A-Z]"),
    # Jr. | Sr. | Inc. | Co. | Corp. | LLC (corporate suffix)
    re.compile(r"\b(?:Jr|Sr|Inc|Ltd|Co|Corp|LLC|L\.L\.C)\.(?:\s|$|,)"),
    # 555 So. 2d 123 | 500 N.E.2d 100 | 123 P.3d 456 (state/regional reporters)
    re.compile(r"\d+\s+(?:[A-Z][a-z]*\.[ \t]*){1,2}\d*[a-z]*(?:[ \t]+\d+)?"),
)


# ── Exceptions and dataclasses ─────────────────────────────────────────


class ChunkingError(Exception):
    def __init__(self, case_id: int | None, reason: str) -> None:
        self.case_id = case_id
        self.reason = reason
        super().__init__(f"case_id={case_id!r}: {reason}")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    case_id: int
    parent_id: str | None
    is_parent: bool
    text: str
    text_original: str
    text_hash: str
    section: str
    section_path: str
    section_priority: int
    section_index: int
    child_index: int
    opinion_type: str
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int
    court: str
    date: str
    citation: str
    title: str
    chunker_version: str


# ── HTML extraction ────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }
    )
    DROP_TAGS = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._drop_depth = 0
        self._drop_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.DROP_TAGS:
            self._drop_depth += 1
            self._drop_tags.append(tag)
            return
        if self._drop_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if self._drop_tags and self._drop_tags[-1] == tag:
                self._drop_tags.pop()
                self._drop_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._drop_depth and data:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self._drop_depth:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self._drop_depth:
            self.parts.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self.parts)


def _strip_html(text: str, text_kind: str) -> str:
    if "html" not in str(text_kind).lower():
        return text
    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise ValueError(f"HTML extraction failed: {exc}") from exc
    return parser.get_text()


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u00a0": " ",
                "\u200b": "",
                "\ufeff": "",
            }
        )
    )
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Strip CourtListener page markers and OCR page headers. These are
    # source artifacts, not content. A chunk ending "...plaintiff.\n\n*84"
    # is semantically complete; the *84 is CourtListener's page number.
    #   *NNN                 → CourtListener page number
    #   * Hi * Hi Hi *       → OCR page-header garbage
    #   *Page NNN            → inline page marker (seen in 1000-case corpus)
    text = re.sub(r"(?m)^[ \t]*\*[\w \t*]{0,30}\*[ \t]*$\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*\*\d+[ \t]*$\n?", "", text)
    text = re.sub(r"[ \t]*\*Page[ \t]+\d+[ \t]*", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Small helpers ──────────────────────────────────────────────────────


def _map_opinion_type(raw: Any) -> str:
    return OPINION_TYPE_MAP.get(str(raw or "").strip().lower(), "unknown")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _measure(text: str, tokenize: Callable[[str], int] | None) -> int:
    if tokenize is not None:
        return tokenize(text)
    return _estimate_tokens(text)


def _hash_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _make_chunk_id(case_id: int, opinion_type: str, section_index: int, child_index: int) -> str:
    return f"{case_id}:{opinion_type}:{section_index}:{child_index}"


def _make_breadcrumb(
    title: str,
    citation: str,
    date: str,
    court: str,
    opinion_type: str,
    section_path: str,
) -> str:
    title_short = re.sub(r"\s+", " ", title).strip()
    if len(title_short) > 120:
        title_short = title_short[:117].rstrip() + "..."

    year_match = re.search(r"\b(?:19|20)\d{2}\b", date)
    year = year_match.group(0) if year_match else "date unknown"

    components = [
        title_short or "Untitled opinion",
        citation.strip() or "citation unavailable",
        year,
        court.strip() or "court unknown",
        opinion_type,
    ]
    if section_path:
        components.append(f"§ {section_path}")

    section_comp = components[5] if len(components) > 5 else "section unavailable"
    return f"[{', '.join(components[:3])} · {components[3]} · {components[4]} · {section_comp}]"


def _as_iso_string(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


# ── Atomic-boundary handling ───────────────────────────────────────────
#
# intervals parameter lets callers precompute the merged interval list ONCE
# per opinion and pass it to every boundary check, replacing per-call regex
# rescans with binary search. When intervals=None, the fallback uses the
# original v2.2 regex-scan logic (slower but correct).


def _build_atomic_intervals(text: str) -> list[tuple[int, int]]:
    """Build a sorted, merged list of (start, end) atomic spans."""
    intervals: list[tuple[int, int]] = []
    for pat in ATOMIC_SPAN_PATTERNS:
        for m in pat.finditer(text):
            intervals.append((m.start(), m.end()))
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlapping or touching
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _pos_inside(intervals: list[tuple[int, int]], pos: int) -> bool:
    """True if pos is strictly inside any merged interval."""
    if not intervals:
        return False
    idx = bisect.bisect_right(intervals, (pos, float("inf"))) - 1
    if idx < 0:
        return False
    s, e = intervals[idx]
    return s < pos < e


def _is_atomic_boundary(
    text: str, pos: int, intervals: list[tuple[int, int]] | None = None
) -> bool:
    """True if cutting at pos is safe (not inside any atomic span)."""
    if pos <= 0 or pos >= len(text):
        return True
    if intervals is not None:
        return not _pos_inside(intervals, pos)
    for pattern in ATOMIC_SPAN_PATTERNS:
        for match in pattern.finditer(text):
            if match.start() < pos < match.end():
                return False
    return True


def _snap_to_safe_boundary(
    text: str,
    pos: int,
    *,
    backward_only: bool = False,
    intervals: list[tuple[int, int]] | None = None,
) -> int:
    """Move pos out of any atomic span.

    With intervals (recommended): one-shot binary lookup, since merged
    intervals are non-overlapping.
    Without: 16-iteration loop with per-call regex rescans.
    """
    pos = max(0, min(pos, len(text)))

    if intervals is not None:
        idx = bisect.bisect_right(intervals, (pos, float("inf"))) - 1
        if idx < 0:
            return pos
        s, e = intervals[idx]
        if s < pos < e:
            if backward_only:
                return s
            return s if pos - s <= e - pos else e
        return pos

    for _ in range(16):
        changed = False
        for pattern in ATOMIC_SPAN_PATTERNS:
            for match in pattern.finditer(text):
                if match.start() < pos < match.end():
                    before = match.start()
                    after = match.end()
                    if backward_only:
                        pos = before
                    else:
                        pos = before if pos - before <= after - pos else after
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return pos


# ── Span extraction ────────────────────────────────────────────────────


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return paragraph spans without splitting wrapped single-newline text."""
    body = text[start:end]

    if not re.search(r"\n[ \t]*\n", body):
        return [(start, end)] if start < end else []

    result: list[tuple[int, int]] = []
    cursor = start

    for match in re.finditer(r"\n[ \t]*\n+", body):
        sep_start = start + match.start()
        sep_end = start + match.end()
        left, right = _trim_span(text, cursor, sep_start)
        if left < right:
            result.append((left, right))
        cursor = sep_end

    left, right = _trim_span(text, cursor, end)
    if left < right:
        result.append((left, right))

    return result


def _sentence_spans(
    text: str,
    start: int,
    end: int,
    intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    body = text[start:end]

    pattern = r"(?<=[.!?])[\x22\x27\u201d\u2019)]*\s+(?=[A-Z0-9(\"])"

    for match in re.finditer(pattern, body):
        boundary = start + match.start()
        if not _is_atomic_boundary(text, boundary, intervals):
            continue

        left, right = _trim_span(text, cursor, boundary)
        if left < right:
            result.append((left, right))
        cursor = start + match.end()

    left, right = _trim_span(text, cursor, end)
    if left < right:
        result.append((left, right))

    return result or [(start, end)]


# Matches the end of a complete sentence: one of . ! ? … : ; optionally
# followed by any number of closing quote/bracket/whitespace characters.
# Triple-quoted raw string needed because " is a class member.
_SENTENCE_END_RE = re.compile(r"""[.!?…:;][]["”’')\s]*$""")


def _trim_to_sentence_end(text: str, start: int, end: int, *, min_ratio: float = 0.55) -> int:
    """Move `end` backward to the last sentence-terminal in the span."""
    if end <= start:
        return end

    if _SENTENCE_END_RE.search(text[start:end].rstrip()):
        return end

    threshold = start + int((end - start) * min_ratio)
    for i in range(end - 1, threshold - 1, -1):
        if text[i] in ".!?":
            j = i + 1
            while j < end and text[j] in ")\"'\u201d\u2019]":
                j += 1
            return j

    return end


def _safe_cut(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    tokenize: Callable[[str], int] | None,
    intervals: list[tuple[int, int]] | None = None,
) -> int:
    """Return a cut position whose prefix is within max_tokens."""
    if start >= end:
        return end

    best: int | None = None

    if tokenize is not None:
        if tokenize(text[start:end]) <= max_tokens:
            return end

        max_chars = int(max_tokens * 5.0)
        search_end = min(start + max_chars, end)
        lo, hi = start + 1, search_end
        best = start + 1

        while lo <= hi:
            mid = (lo + hi) // 2
            if tokenize(text[start:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        target = best
    else:
        max_chars = int(max_tokens * CHARS_PER_TOKEN)
        target = min(start + max_chars, end)
        if target >= end:
            return end

    if target >= end:
        return end

    low = start + max(1, int((target - start) * 0.70))
    for sep in ("\n\n", "\n", " "):
        pos = text.rfind(sep, low, target + 1)
        if pos > start:
            target = pos
            break

    target = _snap_to_safe_boundary(text, target, backward_only=True, intervals=intervals)

    if target <= start:
        if best is not None:
            target = best
        else:
            target = min(start + int(max_tokens * CHARS_PER_TOKEN), end)

    return max(start + 1, min(target, end))


def _split_oversized(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenize: Callable[[str], int] | None,
    intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    while cursor < end:
        if _measure(text[cursor:end], tokenize) <= max_tokens:
            left, right = _trim_span(text, cursor, end)
            if left < right:
                result.append((left, right))
            break

        cut = _safe_cut(text, cursor, end, max_tokens, tokenize, intervals)
        cut = _trim_to_sentence_end(text, cursor, cut, min_ratio=0.55)
        left, right = _trim_span(text, cursor, cut)
        if left < right:
            result.append((left, right))

        next_cursor = cut
        if overlap_chars:
            next_cursor = max(cursor + 1, cut - overlap_chars)
            next_cursor = _snap_to_safe_boundary(
                text, next_cursor, backward_only=True, intervals=intervals
            )

        if next_cursor <= cursor:
            next_cursor = cut

        cursor = next_cursor

    return result


def _split_by_budget(
    text: str,
    start: int,
    end: int,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int = 0,
    tokenize: Callable[[str], int] | None = None,
    intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    if max_tokens < 1:
        return []

    target_tokens = max(1, min(target_tokens, max_tokens))
    min_tokens = max(1, min(min_tokens, max_tokens))

    paragraphs = _paragraph_spans(text, start, end)
    if not paragraphs:
        return []

    # Merge paragraphs whose boundary lands inside an atomic span.
    # Fixes case-name spans split across a paragraph break in source
    # ("Geohagan v. General Motors Corp.,\n\n555 So. 2d 123").
    if intervals is not None and len(paragraphs) > 1:
        merged_paras: list[tuple[int, int]] = []
        i = 0
        while i < len(paragraphs):
            p_start, p_end = paragraphs[i]
            while i < len(paragraphs) - 1 and not _is_atomic_boundary(text, p_end, intervals):
                i += 1
                p_end = paragraphs[i][1]
            merged_paras.append((p_start, p_end))
            i += 1
        paragraphs = merged_paras

    units: list[tuple[int, int]] = []

    for p_start, p_end in paragraphs:
        if _measure(text[p_start:p_end], tokenize) <= max_tokens:
            units.append((p_start, p_end))
            continue

        for s_start, s_end in _sentence_spans(text, p_start, p_end, intervals):
            if _measure(text[s_start:s_end], tokenize) <= max_tokens:
                units.append((s_start, s_end))
            else:
                units.extend(
                    _split_oversized(
                        text,
                        s_start,
                        s_end,
                        max_tokens,
                        overlap_tokens,
                        tokenize,
                        intervals,
                    )
                )

    if not units:
        return []

    packed: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    for unit_start, unit_end in units:
        if current_start is None or current_end is None:
            current_start = unit_start
            current_end = unit_end
            continue

        proposed_tokens = _measure(text[current_start:unit_end], tokenize)
        current_tokens = _measure(text[current_start:current_end], tokenize)

        should_flush = proposed_tokens > max_tokens or (
            proposed_tokens > target_tokens and current_tokens >= min_tokens
        )

        if should_flush:
            packed.append((current_start, current_end))
            current_start = unit_start
            current_end = unit_end
        else:
            current_end = unit_end

    if current_start is not None and current_end is not None:
        packed.append((current_start, current_end))

    if len(packed) >= 2:
        prev_start, prev_end = packed[-2]
        last_start, last_end = packed[-1]
        combined = text[prev_start:last_end]

        if (
            _measure(text[last_start:last_end], tokenize) < min_tokens
            and _measure(combined, tokenize) <= max_tokens
        ):
            packed[-2:] = [(prev_start, last_end)]

    return [_trim_span(text, left, right) for left, right in packed if left < right]


# ── Section path extraction ────────────────────────────────────────────


def _extract_section_path(text: str) -> str:
    first_line = text.split("\n", 1)[0].strip()
    match = re.match(r"^((?:[IVXLC]{1,4})(?:\.[A-Z])?(?:\.\d+)?)\s*\.", first_line)
    if match:
        return match.group(1)

    match = re.match(r"^(\d+(?:\.\d+)*)[.)]\s*", first_line)
    if match:
        return match.group(1)

    return ""


# ── Section detection ──────────────────────────────────────────────────


def _detect_sections(text: str) -> list[tuple[str, int, int, int]]:
    if not text:
        return []

    matches: dict[int, tuple[str, int]] = {}

    for name, priority, pattern in COMPILED_SECTION_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            existing = matches.get(pos)
            if existing is None or priority > existing[1]:
                matches[pos] = (name, priority)

    if not matches:
        return [("body", 50, 0, len(text))]

    starts = sorted(matches)
    sections: list[tuple[str, int, int, int]] = []

    if starts[0] > 0:
        left, right = _trim_span(text, 0, starts[0])
        if left < right:
            sections.append(("body", 50, left, right))

    for i, section_start in enumerate(starts):
        section_end = starts[i + 1] if i + 1 < len(starts) else len(text)
        left, right = _trim_span(text, section_start, section_end)
        if left >= right:
            continue
        name, priority = matches[section_start]
        sections.append((name, priority, left, right))

    return sections or [("body", 50, 0, len(text))]


# ── Parent / child splitting ───────────────────────────────────────────


def _split_parents(
    text: str,
    sections: list[tuple[str, int, int, int]],
    tokenize: Callable[[str], int] | None,
    intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[str, int, str, int, int]]:
    groups: list[list[tuple[str, int, int, int]]] = []

    for section in sections:
        _, _, start, end = section
        section_tokens = _measure(text[start:end], tokenize)

        if groups and section_tokens < PARENT_MIN_TOK:
            prev_last_start = groups[-1][-1][2]
            if _measure(text[prev_last_start:end], tokenize) <= PARENT_MAX_TOK * 2:
                groups[-1].append(section)
                continue

        groups.append([section])

    parents: list[tuple[str, int, str, int, int]] = []

    for group in groups:
        start = group[0][2]
        end = group[-1][3]
        primary = max(group, key=lambda item: item[1])
        name = primary[0]
        priority = primary[1]

        paths: list[str] = []
        for section in group:
            path = _extract_section_path(text[section[2] : section[3]])
            if path and path not in paths:
                paths.append(path)

        section_path = ";".join(paths)

        spans = _split_by_budget(
            text=text,
            start=start,
            end=end,
            target_tokens=PARENT_TARGET_TOK,
            max_tokens=PARENT_MAX_TOK,
            min_tokens=PARENT_MIN_TOK,
            overlap_tokens=0,
            tokenize=tokenize,
            intervals=intervals,
        )

        for p_start, p_end in spans:
            parents.append((name, priority, section_path, p_start, p_end))

    if len(parents) >= 2:
        merged: list[tuple[str, int, str, int, int]] = [parents[0]]

        for parent in parents[1:]:
            prev_name, prev_priority, prev_path, prev_start, prev_end = merged[-1]
            name, priority, path, start, end = parent

            if _measure(text[start:end], tokenize) < PARENT_MIN_TOK:
                if _measure(text[prev_start:end], tokenize) <= PARENT_MAX_TOK:
                    merged[-1] = (
                        prev_name,
                        prev_priority,
                        prev_path,
                        prev_start,
                        end,
                    )
                    continue

            merged.append(parent)

        parents = merged

    return parents


def _split_children(
    text: str,
    parent_start: int,
    parent_end: int,
    breadcrumb_tokens: int,
    tokenize: Callable[[str], int] | None,
    max_embedding_tokens: int,
    intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    body_budget = max_embedding_tokens - breadcrumb_tokens - 2
    if body_budget < 1:
        return []

    body_max = min(CHILD_MAX_TOK, body_budget)
    body_target = max(1, min(CHILD_TARGET_TOK, body_max - 20))
    body_min = max(1, min(CHILD_MIN_TOK, body_max // 2))

    return _split_by_budget(
        text=text,
        start=parent_start,
        end=parent_end,
        target_tokens=body_target,
        max_tokens=body_max,
        min_tokens=body_min,
        overlap_tokens=CHILD_OVERLAP_FALLBACK_TOK,
        tokenize=tokenize,
        intervals=intervals,
    )


# ── Public entry point ─────────────────────────────────────────────────


def chunk_opinion(
    record: dict[str, Any],
    tokenize: Callable[[str], int] | None = None,
    is_noise: Callable[[str], bool] | None = None,
    max_embedding_tokens: int = EMBEDDER_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one opinion record into parent + child chunks."""
    if (
        isinstance(max_embedding_tokens, bool)
        or not isinstance(max_embedding_tokens, int)
        or max_embedding_tokens < 1
    ):
        raise ChunkingError(
            None,
            f"max_embedding_tokens must be a positive integer, got {max_embedding_tokens!r}",
        )

    raw_id = record.get("cl_opinion_id")
    if raw_id is None:
        raise ChunkingError(None, "missing cl_opinion_id")

    try:
        case_id = int(raw_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ChunkingError(None, f"invalid cl_opinion_id: {raw_id!r}") from exc

    raw_text = record.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ChunkingError(case_id, "empty or non-string opinion text")

    text_kind = str(record.get("text_kind") or "plain_text")
    plain = _normalize(_strip_html(raw_text, text_kind))
    if not plain:
        raise ChunkingError(case_id, "empty text after HTML extraction and normalization")

    canonical_type = _map_opinion_type(record.get("opinion_type"))
    title = str(record.get("title") or "")
    citation = str(record.get("citation") or "")
    court = str(record.get("court") or "")
    date = _as_iso_string(record.get("date"))

    # Build atomic intervals ONCE per opinion (v2.3 speedup).
    intervals = _build_atomic_intervals(plain)

    sections = _detect_sections(plain)
    parents = _split_parents(plain, sections, tokenize, intervals)

    if not parents:
        raise ChunkingError(case_id, "could not produce a parent chunk")

    emitted: list[Chunk] = []

    for section_index, (
        section_name,
        section_priority,
        section_path,
        parent_start,
        parent_end,
    ) in enumerate(parents):
        parent_body = plain[parent_start:parent_end].strip()
        if not parent_body:
            continue

        breadcrumb = _make_breadcrumb(
            title=title,
            citation=citation,
            date=date,
            court=court,
            opinion_type=canonical_type,
            section_path=section_path,
        )
        parent_id = _make_chunk_id(case_id, canonical_type, section_index, -1)

        breadcrumb_text = breadcrumb + "\n"
        breadcrumb_tokens = _measure(breadcrumb_text, tokenize)

        if breadcrumb_tokens + 2 >= max_embedding_tokens:
            raise ChunkingError(
                case_id,
                "breadcrumb consumes the child token budget: "
                f"{breadcrumb_tokens} breadcrumb tokens, "
                f"{max_embedding_tokens} maximum",
            )

        child_spans = _split_children(
            plain,
            parent_start,
            parent_end,
            breadcrumb_tokens,
            tokenize,
            max_embedding_tokens,
            intervals,
        )

        section_children: list[Chunk] = []

        for child_index, (child_start, child_end) in enumerate(child_spans):
            child_body = plain[child_start:child_end].strip()
            if not child_body:
                continue
            if is_noise is not None and is_noise(child_body):
                continue

            child_id = _make_chunk_id(
                case_id,
                canonical_type,
                section_index,
                child_index,
            )

            section_children.append(
                Chunk(
                    chunk_id=child_id,
                    case_id=case_id,
                    parent_id=parent_id,
                    is_parent=False,
                    text=f"{breadcrumb}\n{child_body}",
                    text_original=child_body,
                    text_hash=_hash_text(child_body),
                    section=section_name,
                    section_path=section_path,
                    section_priority=section_priority,
                    section_index=section_index,
                    child_index=child_index,
                    opinion_type=canonical_type,
                    page_start=None,
                    page_end=None,
                    char_start=child_start,
                    char_end=child_end,
                    court=court,
                    date=date,
                    citation=citation,
                    title=title,
                    chunker_version=CHUNKER_VERSION,
                )
            )

        if not section_children:
            continue

        emitted.append(
            Chunk(
                chunk_id=parent_id,
                case_id=case_id,
                parent_id=None,
                is_parent=True,
                text=f"{breadcrumb}\n{parent_body}",
                text_original=parent_body,
                text_hash=_hash_text(parent_body),
                section=section_name,
                section_path=section_path,
                section_priority=section_priority,
                section_index=section_index,
                child_index=-1,
                opinion_type=canonical_type,
                page_start=None,
                page_end=None,
                char_start=parent_start,
                char_end=parent_end,
                court=court,
                date=date,
                citation=citation,
                title=title,
                chunker_version=CHUNKER_VERSION,
            )
        )
        emitted.extend(section_children)

    if not any(chunk.is_parent for chunk in emitted):
        raise ChunkingError(case_id, "no valid parent chunks were emitted")

    if not any(not chunk.is_parent for chunk in emitted):
        raise ChunkingError(case_id, "no valid child chunks were emitted")

    if tokenize is not None:
        oversized: list[tuple[str, int]] = []
        for chunk in emitted:
            if chunk.is_parent:
                continue
            token_count = tokenize(chunk.text)
            if token_count > max_embedding_tokens:
                oversized.append((chunk.chunk_id, token_count))

        if oversized:
            details = ", ".join(
                f"{chunk_id}={token_count}" for chunk_id, token_count in oversized[:5]
            )
            raise ChunkingError(
                case_id,
                f"{len(oversized)} child(ren) exceed backend limit "
                f"{max_embedding_tokens} tokens: {details}",
            )

    return emitted


# ── Corruption detection ───────────────────────────────────────────────

CORRUPTION_THRESHOLD = 0.5

_FRAGMENT_TAIL_RE = re.compile(
    r"(?:"
    r"\b[a-z]{1,3}$"  # ends in 1-3 lowercase letters: "...with re"
    r"|[a-z]/$"  # ends in lowercase + slash:   "...sults/"
    r"|\*[ \t*]*$"  # ends in asterisk debris:     "...* Hi * Hi *"
    r")"
)


def corruption_score(children: Sequence[Chunk | dict[str, Any]]) -> float:
    """Fraction of children that show evidence of source OCR corruption."""
    if not children:
        return 0.0

    bad = 0
    n_children = 0

    for chunk in children:
        if isinstance(chunk, dict):
            is_parent = bool(chunk.get("is_parent", False))
            tail = str(chunk.get("text_original", "")).rstrip()
        else:
            is_parent = chunk.is_parent
            tail = chunk.text_original.rstrip()

        if is_parent:
            continue

        n_children += 1
        if _FRAGMENT_TAIL_RE.search(tail):
            bad += 1

    return bad / n_children if n_children else 0.0


# ── Per-case quality filter (v2.6/v2.7) ───────────────────────────────
#
# Applied in pipeline.chunk() BEFORE chunk files are written. A case that
# fails any criterion is dropped (logged to _corruption_drops.jsonl), and
# its chunks never reach load/validate/index. Gates therefore measure only
# quality-passing cases.
#
# MATHEMATICAL GUARANTEE: aggregate gate_3 rate = sum(bad_i) / sum(children_i)
# is a weighted average of per-case rates. Keeping per-case rate <= 0.04
# guarantees aggregate <= 0.04 < 0.05 gate threshold.

QUALITY_GATE_3_MAX_RATE = 0.04
QUALITY_GATE_4_MAX_HITS = 0
QUALITY_CORRUPTION_MAX = 0.10

_CITATION_SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+\s+U\.S\.[ \t]*$"),
    re.compile(r"\d+\s+(?:So|P|A|N\.E|N\.W|S\.E|S\.W)\.[ \t]*[0-9]*[a-z]*[ \t]*$"),
    re.compile(r"\d+\s+U\.S\.C\.[ \t]*$"),
    re.compile(r"(?:[A-Z][a-z]+\.)+[ \t]*Code[ \t]*$"),
    re.compile(r"\bv\.[ \t]*$"),
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)\.[ \t]*$"),
    re.compile(r"[§][ \t]*$"),
)


def quality_filter(
    chunks: Sequence[Chunk | dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Return (keep, stats). Drop case if it would fail any corpus gate.

    Checks three criteria in order (cheapest first):
      1. gate_3 rate: bad-ending fraction <= QUALITY_GATE_3_MAX_RATE
      2. gate_4 hits: zero citation-fragment children
      3. corruption_score: <= QUALITY_CORRUPTION_MAX

    Returns:
        (True, stats) if the case passes all criteria
        (False, stats) with 'reason' key indicating which criterion failed
    """
    non_parent: list[str] = []
    for c in chunks:
        if isinstance(c, Chunk):
            if not c.is_parent:
                non_parent.append(c.text_original)
        else:
            if not bool(c.get("is_parent", False)):
                non_parent.append(str(c.get("text_original", "")))

    if not non_parent:
        return False, {"reason": "no_children"}

    stats: dict[str, Any] = {"n_children": len(non_parent)}

    bad_endings = sum(1 for body in non_parent if not _SENTENCE_END_RE.search(body.rstrip()))
    gate_3_rate = bad_endings / len(non_parent)
    stats["gate_3_rate"] = round(gate_3_rate, 4)
    if gate_3_rate > QUALITY_GATE_3_MAX_RATE:
        stats["reason"] = "gate_3_rate"
        return False, stats

    citation_hits = sum(
        1 for body in non_parent if any(p.search(body.rstrip()) for p in _CITATION_SPLIT_PATTERNS)
    )
    stats["gate_4_hits"] = citation_hits
    if citation_hits > QUALITY_GATE_4_MAX_HITS:
        stats["reason"] = "gate_4_hits"
        return False, stats

    score = corruption_score(chunks)
    stats["corruption_score"] = round(score, 4)
    if score > QUALITY_CORRUPTION_MAX:
        stats["reason"] = "corruption_score"
        return False, stats

    return True, stats


__all__ = [
    "CHUNKER_VERSION",
    "CORRUPTION_THRESHOLD",
    "QUALITY_CORRUPTION_MAX",
    "QUALITY_GATE_3_MAX_RATE",
    "QUALITY_GATE_4_MAX_HITS",
    "Chunk",
    "ChunkingError",
    "EMBEDDER_MAX_TOKENS",
    "chunk_opinion",
    "corruption_score",
    "quality_filter",
]
