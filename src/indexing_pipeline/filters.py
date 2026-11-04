"""Text and metadata quality filters for the indexing pipeline.

Pure functions, no I/O, no model, no database. Deterministic: same input
always produces the same output.

Two responsibilities:
  1. Metadata normalization — clean CourtListener cluster fields
  2. Noise detection — reject text snippets that are provably noise

Grounding (measured by inspect_data.py on the min-scale corpus):
  - 'precedential_status' contains '2' (3 rows), '5' (2 rows),
    ' Asst. Atty. Gen.' (2 rows), and nan (51 rows). These are CSV
    field-shift artifacts, not real statuses. Silently coercing them to
    'Unpublished' would misclassify opinions; NULL is honest.
  - 23.3% of paragraphs are <50 chars. Most are page numbers, footnote
    markers, and section headers. Dropping provable noise reduces index
    bloat without affecting retrieval quality.

What this file is NOT:
  - Not a deduplicator. See dedup.py for cross-case paragraph dedup.
  - Not a chunker. See chunker.py for segmentation.
  - Not a tokenizer. Size checks live in chunker._measure().

Design philosophy: conservative. Rejecting a legitimate short chunk
costs more retrieval quality than keeping a noise chunk. Every reject
must be defensible: a pure-digit string like "2020" is a page number;
a citation like "410 U.S. 113" is not. When in doubt, keep.
"""

from __future__ import annotations

import re
from typing import Any

FILTERS_VERSION = "1"


# ── Metadata normalization ─────────────────────────────────────────────


# CourtListener's documented precedential statuses.
# Source: datasets.md §4.1 (CourtListener opinion-clusters schema).
# Map from lowercased form → canonical title form. Explicit mapping
# (not title-casing) because of hyphens: 'in-chambers' → 'In-chambers',
# not 'In-Chambers'.
CANONICAL_PRECEDENTIAL_STATUSES: dict[str, str] = {
    "published": "Published",
    "unpublished": "Unpublished",
    "errata": "Errata",
    "separate": "Separate",
    "in-chambers": "In-chambers",
    "relating-to": "Relating-to",
}


def normalize_precedential_status(raw: Any) -> str | None:
    """Return the canonical status string, or None if unrecognized.

    precedential_status is a correctness filter: unpublished opinions
    are not citable as precedent in most US jurisdictions. Getting this
    wrong silently excludes opinions, so we never guess — unrecognized
    input becomes None, not a coerced default.

    Returns:
        - Canonical string ('Published', 'Unpublished', ...) if recognized
        - None for unrecognized input (CSV field shifts, nan, empty)
        - None for non-string input (defensive)

    Examples:
        >>> normalize_precedential_status("Published")
        'Published'
        >>> normalize_precedential_status("PUBLISHED")
        'Published'
        >>> normalize_precedential_status("2") is None
        True
        >>> normalize_precedential_status(" Asst. Atty. Gen.") is None
        True
        >>> normalize_precedential_status(float("nan")) is None
        True
    """
    if raw is None:
        return None
    try:
        s = str(raw).strip().lower()
    except Exception:
        return None
    if not s or s == "nan":
        return None
    return CANONICAL_PRECEDENTIAL_STATUSES.get(s)


# ── Noise detection ────────────────────────────────────────────────────


# Precompiled regexes — is_noise_chunk runs once per emitted child,
# so compile once at module load.
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_LOWER_RE = re.compile(r"[a-z]")
_DIGIT_RE = re.compile(r"\d")


def is_noise_chunk(
    text: str,
    *,
    min_alnum_ratio: float = 0.30,
) -> bool:
    """Return True if this text snippet is provable noise.

    "Provable" means: the text carries no retrievable semantic content.
    Used as a child-emit filter inside chunk_opinion — chunks whose
    entire body would be dropped here are never emitted.

    Rejects (returns True):
        - empty or whitespace-only
        - no alphanumeric characters at all ("* * *", "---", "...")
        - pure digits ("123", "2020", "42")
        - very short (< 8 chars) with no lowercase and no digits
          (section headers like "ALASKA", "I.", "U.S.")
        - alphanumeric character ratio below min_alnum_ratio
          (heavily symbol-laden text)

    Keeps (returns False):
        - "410 U.S. 113"
        - "Roe v. Wade, 410 U.S. 113 (1973)"
        - "See generally"
        - "APPEAL FROM THE CIRCUIT COURT"
        - "§ 1983"                      (7 chars, has digits)
        - "REVERSED."                   (9 chars, > 8 threshold)
        - "Affirmed."                   (has lowercase)
        - "Id." / "supra"               (have lowercase)

    Args:
        text:            the chunk body to evaluate (no breadcrumb)
        min_alnum_ratio: minimum fraction of alphanumeric characters.
                         0.30 default. Adjust only with measured evidence.

    Returns:
        True if the text should be dropped, False if it should be kept.

    Raises:
        TypeError: if text is not a string.

    Examples:
        >>> is_noise_chunk("410 U.S. 113")
        False
        >>> is_noise_chunk("* * *")
        True
        >>> is_noise_chunk("2020")
        True
        >>> is_noise_chunk("ALASKA")
        True
        >>> is_noise_chunk("§ 1983")
        False
    """
    if not isinstance(text, str):
        raise TypeError(f"is_noise_chunk expects str, got {type(text).__name__}")

    stripped = text.strip()
    if not stripped:
        return True

    # No alphanumeric characters → no retrievable content.
    if not _ALNUM_RE.search(stripped):
        return True

    # Pure digits (page numbers, footnote markers).
    # Note: stripped is guaranteed non-empty here, so all() is not vacuous.
    if all(ch.isdigit() or ch.isspace() for ch in stripped):
        return True

    # Very short with no lowercase and no digits → section header or
    # stray symbol. len < 8 catches "ALASKA" (6 chars) but not
    # "APPEAL FROM THE CIRCUIT COURT" (30 chars). The "no digits" clause
    # preserves "§ 1983" (7 chars, has digits).
    if len(stripped) < 8 and not _LOWER_RE.search(stripped) and not _DIGIT_RE.search(stripped):
        return True

    # Heavily symbol-laden text. A real legal chunk has mostly letters
    # and digits; ruler-line boilerplate doesn't.
    alnum = sum(1 for ch in stripped if ch.isalnum())
    if alnum / len(stripped) < min_alnum_ratio:
        return True

    return False


__all__ = [
    "FILTERS_VERSION",
    "CANONICAL_PRECEDENTIAL_STATUSES",
    "normalize_precedential_status",
    "is_noise_chunk",
]
