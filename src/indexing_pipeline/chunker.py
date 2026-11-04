"""Pure algorithm — structure-aware chunking of court opinions.

No I/O, no database, no model. Deterministic: same input + same tokenizer
produce the same output.

When a `tokenize` callable is provided, it is used for every size check so
the chunk boundaries match the embedder's real tokenizer exactly. When
omitted, a 3.0 chars/token heuristic is used (safe for citation-dense
legal text).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

# ── Versioning ──────────────────────────────────────────────────────────
# Bumped to v1.3 — atomic-span patterns expanded for case names, state
# codes, honorifics, corporate suffixes, and state reporters. All previously
# chunked output is invalidated.

CHUNKER_VERSION = "chunk-v1.3"


# ── Opinion-type mapping (CourtListener codes → canonical labels) ───────

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
# Observed chars/token on Alabama case law: min 3.13, p50 3.71, p95 4.02.
# 3.0 gives ~5% safety margin below the observed minimum. The production
# path passes a real tokenizer via chunk_opinion(tokenize=...) so this
# heuristic is used only when the caller cannot supply one.

CHARS_PER_TOKEN = 3.0

PARENT_TARGET_TOK = 1000
PARENT_MAX_TOK = 1400
PARENT_MIN_TOK = 500

CHILD_TARGET_TOK = 320
CHILD_MAX_TOK = 400
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
    # ── Added in v1.3 to stop mid-citation and mid-name cuts ───────
    # Case names: "Holyfield v. Moates", "Brown v. Board"
    re.compile(r"\b[A-Z][A-Za-z']+\s+v\.\s+[A-Z][A-Za-z']+"),
    # State code citations: "Ala. Code", "N.Y. Code", "Cal. Civ. Code"
    re.compile(r"\b(?:[A-Z][a-z]*\.)+\s*Code\b"),
    # Honorifics followed by a capitalised name: "Ms. Hilburn", "Dr. Ellis"
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)\.\s+[A-Z]"),
    # Corporate suffixes: "Smith Co.", "ACME Inc.", "John Smith, Jr."
    re.compile(r"\b(?:Jr|Sr|Inc|Ltd|Co|Corp|LLC|L\.L\.C)\.(?:\s|$|,)"),
    # State/regional reporters: "565 So.2d 186", "123 P.3d 456", "500 N.E.2d 100"
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
    text: str  # breadcrumb + body  (for BM25 and display)
    text_original: str  # body only           (for dense embedding + hashing)
    text_hash: str  # sha256 of text_original
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Small helpers ──────────────────────────────────────────────────────


def _map_opinion_type(raw: Any) -> str:
    return OPINION_TYPE_MAP.get(str(raw or "").strip().lower(), "unknown")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _measure(text: str, tokenize: Callable[[str], int] | None) -> int:
    """Real tokenizer when provided, else char-based estimate."""
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
    """Move pos out of any atomic span.

    backward_only=True always moves the cut backward. This is used after
    a size-based cut so snapping cannot push the boundary past the budget.
    """
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


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    body = text[start:end]

    # Primary: double-newline (paragraph) boundaries
    if re.search(r"\n[ \t]*\n", body):
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

    # Fallback: single-newline boundaries (line-level split)
    if "\n" in body:
        result = []
        cursor = start
        for match in re.finditer(r"\n+", body):
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

    # Last resort: one span (whole opinion) — sentence splitter handles this
    return [(start, end)] if start < end else []


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    body = text[start:end]
    for match in re.finditer(r'(?<=[.!?])[\x22\x27\u201d\u2019)]*\s+(?=[A-Z0-9("])', body):
        boundary = start + match.start()
        # Skip boundaries that fall inside an atomic span
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


def _safe_cut(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    tokenize: Callable[[str], int] | None,
) -> int:
    """Return the largest position <= start + budget, on a word boundary,
    not inside an atomic span.

    With a real tokenizer, binary-search the largest cut whose measured
    token count is <= max_tokens. Without one, cut at max_tokens ×
    CHARS_PER_TOKEN characters and snap backward to a word boundary.
    """
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    target = min(start + max_chars, end)
    if target >= end:
        return end

    if tokenize is not None:
        lo, hi, best = start + 1, target, start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if tokenize(text[start:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        target = best

    # Snap backward to the nearest paragraph/sentence/word boundary.
    low = start + max(1, int((target - start) * 0.70))
    for sep in ("\n\n", "\n", " "):
        pos = text.rfind(sep, low, target + 1)
        if pos > start:
            target = pos
            break

    # Atomic-boundary snap: backward only, so the size budget is respected.
    target = _snap_to_safe_boundary(text, target, backward_only=True)

    if target <= start or target > end:
        target = min(start + max_chars, end)
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
        left, right = _trim_span(text, cursor, cut)
        if left < right:
            result.append((left, right))
        next_cursor = cut
        if overlap_chars:
            next_cursor = max(cursor + 1, cut - overlap_chars)
            next_cursor = _snap_to_safe_boundary(text, next_cursor)
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
                    _split_oversized(text, s_start, s_end, max_tokens, overlap_tokens, tokenize)
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

    # Merge a small trailing fragment when it fits safely.
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
) -> list[tuple[str, int, str, int, int]]:
    # Group sections: a small (< PARENT_MIN_TOK) section merges into the
    # previous group only if the previous group's LAST section, extended
    # by this one, stays within 2 × PARENT_MAX_TOK. Measuring from the
    # group's FIRST section (the old behaviour) rejected the merge whenever
    # the group already contained a full-size body — which is exactly the
    # case for a short judgment following a long majority opinion.
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

    # Post-pass: merge any residual undersized parent into its predecessor
    # when the combined span fits within PARENT_MAX_TOK. Catches cases where
    # a small section could not merge at the group level because its
    # predecessor was already at the size ceiling.
    if len(parents) >= 2:
        merged: list[tuple[str, int, str, int, int]] = [parents[0]]
        for parent in parents[1:]:
            prev_name, prev_priority, prev_path, prev_start, prev_end = merged[-1]
            name, priority, path, start, end = parent
            if _measure(text[start:end], tokenize) < PARENT_MIN_TOK:
                if _measure(text[prev_start:end], tokenize) <= PARENT_MAX_TOK:
                    merged[-1] = (prev_name, prev_priority, prev_path, prev_start, end)
                    continue
            merged.append(parent)
        parents = merged

    return parents


def _split_children(
    text: str,
    parent_start: int,
    parent_end: int,
    tokenize: Callable[[str], int] | None,
) -> list[tuple[int, int]]:
    return _split_by_budget(
        text=text,
        start=parent_start,
        end=parent_end,
        target_tokens=CHILD_TARGET_TOK,
        max_tokens=CHILD_MAX_TOK,
        min_tokens=CHILD_MIN_TOK,
        overlap_tokens=CHILD_OVERLAP_FALLBACK_TOK,
        tokenize=tokenize,
    )


# ── Public entry point ─────────────────────────────────────────────────


def chunk_opinion(
    record: dict[str, Any],
    tokenize: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk one opinion record into parent + child chunks.

    Args:
        record:   dict with cl_opinion_id, text, text_kind, opinion_type,
                  title, citation, court, date.
        tokenize: optional callable mapping a string to its token count in
                  the embedder's tokenizer. When provided, all size checks
                  use it exactly; when omitted, a 3.0 chars/token heuristic
                  is used.

    Raises:
        ChunkingError: on missing ID, empty text, or no producible chunks.
    """
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
        parent_id = _make_chunk_id(case_id, canonical_type, section_index, -1)

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

        child_spans = _split_children(plain, parent_start, parent_end, tokenize)
        for child_index, (child_start, child_end) in enumerate(child_spans):
            child_body = plain[child_start:child_end].strip()
            if not child_body:
                continue
            child_id = _make_chunk_id(case_id, canonical_type, section_index, child_index)

            emitted.append(
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

    if not any(c.is_parent for c in emitted):
        raise ChunkingError(case_id, "no valid parent chunks were emitted")
    if not any(not c.is_parent for c in emitted):
        raise ChunkingError(case_id, "no valid child chunks were emitted")

    return emitted
