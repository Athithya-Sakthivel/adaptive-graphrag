"""Pure algorithm — structure-aware chunking of court opinions.

No I/O, no database, no model. Deterministic: same input + same tokenizer
+ same is_noise predicate + same token limit produce the same output.

When a `tokenize` callable is provided, it is used for every size check so
the chunk boundaries match the embedder's real tokenizer exactly. When
omitted, a 2.25 chars/token heuristic is used.

v2.2 changes:
  - Tier 1: ATOMIC_SPAN_PATTERNS extended to cover multi-word defendant
    names ("Geohagan v. General Motors Corp.,") and the case-name-then-
    reporter cut ("...Corp., | 555 So. 2d") that the previous pattern
    could not protect. Fixes the 1088271 failure class.
  - Tier 2: corruption_score() + CORRUPTION_THRESHOLD. Provides the
    detection primitive the pipeline uses to identify opinions whose
    source text has reverse-page-break OCR corruption (e.g. "re\n\nsults/").
    Detection only — the pipeline decides whether to drop based on the
    score. Chunker stays pure.

v2.1 changes:
  - _normalize strips CourtListener page markers (*NNN, * Hi * Hi *) at
    ingest, before paragraph or sentence splitting. Fixes the class of
    chunk whose tail was "...plaintiff.\n\n*84".
  - _SENTENCE_END_RE accepts ":" as a terminal. Chunks ending in a
    block-quote introducer ("...from Dr. Ussery's testimony :") are
    complete sentences, not fragments.

v2.0 changes:
  - Single newlines are no longer treated as paragraph boundaries.
    Wrapped lines remain together; sentence splitting handles them.
  - `max_embedding_tokens` is configurable per backend. Defaults to 464
    for OpenVINO; callers may pass 512 for a verified PyTorch backend.
  - Child-body budget is calculated from the actual breadcrumb size.
    It never forces a minimum body size that exceeds the remaining budget.
  - Post-emit validation checks against the selected backend limit.

v1.5 changes:
  - Optional `is_noise` callable on chunk_opinion. Applied to child bodies
    only (not parents or breadcrumbs). Children identified as noise
    are not emitted.
  - Parent is emitted only if at least one child survives noise filtering.

v1.4 changes:
  - `_safe_cut` uses binary search whenever a tokenizer is provided.
  - Breadcrumb headroom is reserved in the child token budget.
  - Post-emit validation rejects children exceeding the token limit.
  - CHARS_PER_TOKEN reduced from 3.0 to 2.25.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

# ── Versioning ──────────────────────────────────────────────────────────

CHUNKER_VERSION = "chunk-v2.2"


# ── Default embedder token limit ────────────────────────────────────────
#
# 464 is the current OpenVINO compiled sequence limit.
# It is a default, not a universal transformer limit. Callers using a
# verified backend with a different sequence length must pass that limit
# through max_embedding_tokens to chunk_opinion().

EMBEDDER_MAX_TOKENS = 464


# ── Opinion-type mapping (CourtListener codes -> canonical labels) ──────

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

# Observed chars/token on the target corpus with the dinghy tokenizer:
# min 2.0, p05 2.25, p50 4.04. Dense citation blocks are the worst case.
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
    re.compile(r"\d+\s+[A-Z][A-Za-z.']*\s+\d+"),  # 410 U.S. 113
    re.compile(r"\d+\s+U\.S\.C\.\s+§+\s*\d+[\w().]*"),  # 42 U.S.C. § 1983
    re.compile(r"§+\s*\d+[\w().-]*"),  # § 1983
    re.compile(r"No\.\s*\d{2}-\d+"),  # No. 12-345
    re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?"),  # $10,000
    re.compile(r"\b(?:19|20)\d{2}\b"),  # 1973
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # 06/15/2020
    # Basic case name: single plaintiff word, single defendant word.
    # Fast path for "Roe v. Wade" style citations.
    re.compile(r"\b[A-Z][A-Za-z']+\s+v\.\s+[A-Z][A-Za-z']+"),
    # Case name with corporate suffix in plaintiff ("Safeguard Business
    # Sys., Inc. v. Smith"): allows punctuation inside plaintiff words.
    re.compile(r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+[A-Z]"),
    # Extended case name — up to 3 more words after the defendant's first
    # word. Fixes "Geohagan v. General Motors Corp." (v2.2). Catches cuts
    # INSIDE a multi-word defendant name even when no reporter follows.
    re.compile(
        r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+"
        r"[A-Z][\w.,'\-]*\.?"
        r"(?:(?<=\w)[ \t]+[A-Z][\w.,'\-]*\.?){0,3}"
    ),
    # Case name + reporter — bridges the cut between "...Corp.," and
    # "555 So. 2d". The signature is "case name, reporter volume", i.e.
    # comma, whitespace, digits, whitespace, capital. Fixes the case
    # 1088271 failure where the cut landed at "Corp., | 555 So. 2d".
    re.compile(
        r"\b(?:[A-Z][\w.,'\-]*[ \t]+){1,6}v\.[ \t]+"
        r"[A-Z][\w.,'\-]*\.?"
        r"(?:(?<=\w)[ \t]+[A-Z][\w.,'\-]*\.?){0,3}"
        r",\s+\d+\s+[A-Z]"
    ),
    re.compile(r"\b(?:[A-Z][a-z]*\.)+\s*Code\b"),  # Ala. Code
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)\.\s+[A-Z]"),
    re.compile(r"\b(?:Jr|Sr|Inc|Ltd|Co|Corp|LLC|L\.L\.C)\.(?:\s|$|,)"),
    re.compile(r"\d+\s+(?:So|P|A|N\.E|N\.W|S\.E|S\.W)\.\s*\d*[a-z]*\s+\d+"),
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
    # Only standalone lines of ≤30 chars matching ^\*...\*$ are removed.
    text = re.sub(r"(?m)^[ \t]*\*[\w \t*]{0,30}\*[ \t]*$\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*\*\d+[ \t]*$\n?", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Small helpers ──────────────────────────────────────────────────────


def _map_opinion_type(raw: Any) -> str:
    return OPINION_TYPE_MAP.get(str(raw or "").strip().lower(), "unknown")


def _estimate_tokens(text: str) -> int:
    """Estimate tokens for the no-tokenizer fallback path."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _measure(text: str, tokenize: Callable[[str], int] | None) -> int:
    if tokenize is not None:
        return tokenize(text)
    return _estimate_tokens(text)


def _hash_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _make_chunk_id(
    case_id: int,
    opinion_type: str,
    section_index: int,
    child_index: int,
) -> str:
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


def _is_atomic_boundary(text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(text):
        return True
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
) -> int:
    """Move pos out of any atomic span."""
    pos = max(0, min(pos, len(text)))
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


def _paragraph_spans(
    text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Return paragraph spans without splitting wrapped single-newline text.

    Double newlines are treated as paragraph separators. A single newline
    is not sufficient evidence of a paragraph boundary: legal opinions
    frequently contain hard-wrapped lines and line breaks inserted by
    source extraction. Those lines remain in one span and are handled by
    the sentence splitter or token-budget splitter.
    """
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
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    body = text[start:end]

    pattern = r"(?<=[.!?])[\x22\x27\u201d\u2019)]*\s+(?=[A-Z0-9(\"])"

    for match in re.finditer(pattern, body):
        boundary = start + match.start()
        if not _is_atomic_boundary(text, boundary):
            continue

        left, right = _trim_span(text, cursor, boundary)
        if left < right:
            result.append((left, right))
        cursor = start + match.end()

    left, right = _trim_span(text, cursor, end)
    if left < right:
        result.append((left, right))

    return result or [(start, end)]


# Matches the end of a complete sentence: one of . ! ? … : optionally
# followed by closing quotes/brackets. Used by _trim_to_sentence_end to
# decide whether a chunk already ends cleanly.
_SENTENCE_END_RE = re.compile(r'[.!?…:](?:["”’)\]]+)?$')


def _trim_to_sentence_end(
    text: str,
    start: int,
    end: int,
    *,
    min_ratio: float = 0.55,
) -> int:
    """Move `end` backward to the last sentence-terminal in the span.

    Used by `_split_oversized` after a size-based cut. A chunk that ends
    mid-sentence is semantically worse than one that ends a few tokens
    short of the budget: the reader (and the embedder) sees an incomplete
    thought.

    Returns `end` unchanged when:
      - the span already ends in punctuation, OR
      - no sentence terminal exists within the last (1 - min_ratio) of
        the span (avoids shrinking to near-empty just to hit a period).

    `min_ratio` (0.55) means the trimmed chunk must retain at least 55%
    of the intended size. Tuned empirically: 0.55 clears the <=2%
    boundary-integrity gate without materially reducing effective chunk
    size.
    """
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

    target = _snap_to_safe_boundary(
        text,
        target,
        backward_only=True,
    )

    if target <= start:
        if best is not None:
            target = best
        else:
            target = min(
                start + int(max_tokens * CHARS_PER_TOKEN),
                end,
            )

    return max(start + 1, min(target, end))


def _split_oversized(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenize: Callable[[str], int] | None,
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

        cut = _safe_cut(text, cursor, end, max_tokens, tokenize)
        # Prefer a clean sentence boundary if one is nearby.
        cut = _trim_to_sentence_end(text, cursor, cut, min_ratio=0.55)
        left, right = _trim_span(text, cursor, cut)
        if left < right:
            result.append((left, right))

        next_cursor = cut
        if overlap_chars:
            next_cursor = max(cursor + 1, cut - overlap_chars)
            next_cursor = _snap_to_safe_boundary(text, next_cursor, backward_only=True)

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
) -> list[tuple[int, int]]:
    if max_tokens < 1:
        return []

    target_tokens = max(1, min(target_tokens, max_tokens))
    min_tokens = max(1, min(min_tokens, max_tokens))

    paragraphs = _paragraph_spans(text, start, end)
    if not paragraphs:
        return []

    units: list[tuple[int, int]] = []

    for p_start, p_end in paragraphs:
        if _measure(text[p_start:p_end], tokenize) <= max_tokens:
            units.append((p_start, p_end))
            continue

        for s_start, s_end in _sentence_spans(text, p_start, p_end):
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

        proposed_tokens = _measure(
            text[current_start:unit_end],
            tokenize,
        )
        current_tokens = _measure(
            text[current_start:current_end],
            tokenize,
        )

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
    match = re.match(
        r"^((?:[IVXLC]{1,4})(?:\.[A-Z])?(?:\.\d+)?)\s*\.",
        first_line,
    )
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
) -> list[tuple[int, int]]:
    """Split a parent into child bodies within the selected backend limit.

    The child budget reserves the measured breadcrumb size plus two
    additional tokens for boundary/tokenization variation. The available
    body budget is never raised to CHILD_MIN_TOK: it may be smaller when
    the breadcrumb is long or the backend limit is low.
    """
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
    )


# ── Public entry point ─────────────────────────────────────────────────


def chunk_opinion(
    record: dict[str, Any],
    tokenize: Callable[[str], int] | None = None,
    is_noise: Callable[[str], bool] | None = None,
    max_embedding_tokens: int = EMBEDDER_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one opinion record into parent + child chunks.

    Args:
        record: dict with cl_opinion_id, text, text_kind, opinion_type,
            title, citation, court, date.
        tokenize: optional callable mapping text to its token count using
            the embedder's tokenizer. When omitted, a 2.25 chars/token
            estimate is used.
        is_noise: optional pure predicate applied to child bodies only.
            Children for which it returns True are omitted. A parent is
            omitted if all its children are filtered.
        max_embedding_tokens: maximum token count permitted for the full
            child text, including breadcrumb and body. Defaults to 464
            for OpenVINO. Set to another value only when the selected
            backend's actual sequence limit has been verified.

    Raises:
        ChunkingError: on invalid input, no producible chunks, invalid
            token limit, or any emitted child exceeding the selected
            backend limit.
    """
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
        raise ChunkingError(
            None,
            f"invalid cl_opinion_id: {raw_id!r}",
        ) from exc

    raw_text = record.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ChunkingError(case_id, "empty or non-string opinion text")

    text_kind = str(record.get("text_kind") or "plain_text")
    plain = _normalize(_strip_html(raw_text, text_kind))
    if not plain:
        raise ChunkingError(
            case_id,
            "empty text after HTML extraction and normalization",
        )

    canonical_type = _map_opinion_type(record.get("opinion_type"))
    title = str(record.get("title") or "")
    citation = str(record.get("citation") or "")
    court = str(record.get("court") or "")
    date = _as_iso_string(record.get("date"))

    sections = _detect_sections(plain)
    parents = _split_parents(plain, sections, tokenize)

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
        parent_id = _make_chunk_id(
            case_id,
            canonical_type,
            section_index,
            -1,
        )

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
        )

        # Build children first so a parent is never emitted without a
        # surviving child after noise filtering.
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

    # Validate full child text, including breadcrumb, with the actual
    # tokenizer. This is the final guard against backend sequence overflow.
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


# ── Corruption detection (Tier 2) ──────────────────────────────────────
#
# Reverse-page-break OCR corruption leaves chunks ending in short lowercase
# fragments ("...deals with re"), lowercase+slash ("...sults/"), or asterisk
# debris ("...* Hi * Hi Hi *"). These patterns cannot be repaired by any
# boundary logic — the corruption *is* the boundary. The honest choice is
# to detect severely corrupted opinions and let the pipeline drop them,
# with the drop logged to a visible artifact file.
#
# This function returns a score, not a boolean. The pipeline decides
# whether to drop based on CORRUPTION_THRESHOLD. Threshold 0.5 means
# >50% of children must show corruption for the opinion to be dropped —
# a conservative gate that never fires on isolated fragments in otherwise
# clean opinions.

CORRUPTION_THRESHOLD = 0.5

_FRAGMENT_TAIL_RE = re.compile(
    r"(?:"
    r"\b[a-z]{1,3}$"  # ends in 1-3 lowercase letters: "...with re"
    r"|[a-z]/$"  # ends in lowercase + slash:   "...sults/"
    r"|\*[ \t*]*$"  # ends in asterisk debris:     "...* Hi * Hi *"
    r")"
)


def corruption_score(children: Sequence[Chunk | dict[str, Any]]) -> float:
    """Fraction of children that show evidence of source OCR corruption.

    Accepts either Chunk objects (as returned by chunk_opinion) or dicts
    (as written to chunked/*.jsonl by the pipeline). Only non-parent
    chunks are scored — parents are not indexed.

    Returns 0.0 when there are no children.

    Detects three classes of reverse-page-break / OCR debris:
      - ends in a 1-3 letter lowercase fragment ("...deals with re")
      - ends in a lowercase letter followed by slash ("...sults/")
      - ends in asterisk debris ("...* Hi * Hi *")

    A score above CORRUPTION_THRESHOLD means >50% of children are affected
    — the opinion cannot be meaningfully indexed without source-level
    repair. A score below means the corruption is isolated and the
    well-formed majority of chunks are still usable.
    """
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


__all__ = [
    "CHUNKER_VERSION",
    "CORRUPTION_THRESHOLD",
    "Chunk",
    "ChunkingError",
    "EMBEDDER_MAX_TOKENS",
    "chunk_opinion",
    "corruption_score",
]
