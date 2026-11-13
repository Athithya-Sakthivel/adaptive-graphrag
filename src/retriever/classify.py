"""Query classification and reference extraction.

Classification precedence (first match wins):
    1. Explicit opinion ID            -> LOOKUP
    2. Explicit citation              -> LOOKUP
    3. Validity keyword present       -> VALIDITY
    4. Holding keyword present        -> HOLDING
    5. Case reference present         -> LOOKUP
    6. Default                        -> HOLDING

Rules 3 and 4 fire before rule 5 because "has X been overruled" and
"what did the court hold in X" are questions about X, not requests
for the list of cases citing X.
"""

from __future__ import annotations

import re
from enum import StrEnum


class QueryClass(StrEnum):
    HOLDING = "HOLDING"
    LOOKUP = "LOOKUP"
    VALIDITY = "VALIDITY"


_CITATION_PATTERN = re.compile(
    r"\b\d+[a-z]?\s+"
    r"[A-Z][A-Za-z0-9.]{1,9}"
    r"(?:\s+[A-Z0-9][A-Za-z0-9.]{0,9}){0,2}"
    r"\s+\d+\b",
    re.UNICODE,
)

_OPINION_ID_PATTERN = re.compile(
    r"\bopinion(?:_?id)?\s*[:#]?\s*(\d{1,10})\b",
    re.IGNORECASE,
)

# Case reference patterns - strict capitalization to avoid matching template words
_CASE_REFERENCE_PATTERNS = (
    # Ex parte, In re, Matter of
    re.compile(
        r"\b((?:Ex\s+[Pp]arte|Ex\s+[Rr]el\.?|In\s+[Rr]e|Matter\s+of)\s+"
        r"[A-Z][a-zA-Z.'&,\-]+(?:\s+[A-Za-z][a-zA-Z.'&,\-]+)*)"
    ),
    # Standard X v. Y
    re.compile(
        r"\b([A-Z][a-zA-Z.'&,\-]+(?:\s+[A-Za-z][a-zA-Z.'&,\-]+)*"
        r"\s+[vV]\.?\s+"
        r"[A-Z][a-zA-Z.'&,\-]+(?:\s+[A-Za-z][a-zA-Z.'&,\-]+)*)"
    ),
    # Board of X, People v. X
    re.compile(
        r"\b((?:Board|Department|People|State|Commonwealth|City|Town|County)\s+of\s+"
        r"[A-Z][a-zA-Z.'&,\-]+(?:\s+[A-Za-z][a-zA-Z.'&,\-]+)*)"
    ),
)

_VALIDITY_KEYWORDS = frozenset(
    {
        "overruled",
        "overrule",
        "overruling",
        "overrules",
        "abrogated",
        "abrogate",
        "abrogating",
        "abrogates",
        "reversed",
        "reverse",
        "reversing",
        "reverses",
        "invalidated",
        "invalidate",
        "invalidating",
        "invalidates",
        "superseded",
        "supersede",
        "superseding",
        "supersedes",
        "vacated",
        "vacate",
        "vacating",
        "vacates",
        "still good law",
        "good law",
        "no longer good",
    }
)

_HOLDING_PHRASES = (
    "what did the court hold",
    "what is the holding",
    "what was the holding",
    "holding that",
    "the holding",
    "court held",
    "court holds",
    "court holding",
    "court's holding",
    "rule of law",
    "legal rule",
    "legal standard",
)

# FIX: Added "hold" to catch "We hold that..." patterns
_HOLDING_WORDS_PATTERN = re.compile(r"\b(hold|holding|held|holds)\b", re.IGNORECASE)


def extract_citation(query: str) -> str | None:
    match = _CITATION_PATTERN.search(query)
    return match.group(0).strip() if match else None


def extract_opinion_id(query: str) -> int | None:
    match = _OPINION_ID_PATTERN.search(query)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_case_reference(query: str) -> str | None:
    """Return first case-name reference, stripping query templates.

    Strips common prefixes like "Which cases cite" and suffixes like
    "been overruled" that the regex might greedily capture.
    """
    for pattern in _CASE_REFERENCE_PATTERNS:
        match = pattern.search(query)
        if match is not None:
            extracted = match.group(1).strip()

            # Strip query template prefixes
            for prefix in [
                "Which cases cite ",
                "Has ",
                "Is ",
                "Was ",
                "Did ",
                "Find cases citing ",
            ]:
                if extracted.lower().startswith(prefix.lower()):
                    extracted = extracted[len(prefix) :].strip()

            # Strip query template suffixes
            for suffix in [" been overruled", " overruled", " cited", "?"]:
                if extracted.lower().endswith(suffix.lower()):
                    extracted = extracted[: -len(suffix)].strip()

            return extracted if extracted else None
    return None


def has_case_reference(query: str) -> bool:
    return extract_case_reference(query) is not None


def _contains_validity_keyword(query: str) -> bool:
    lower = query.lower()
    return any(keyword in lower for keyword in _VALIDITY_KEYWORDS)


def _contains_holding_keyword(query: str) -> bool:
    lower = query.lower()
    if any(phrase in lower for phrase in _HOLDING_PHRASES):
        return True
    return _HOLDING_WORDS_PATTERN.search(lower) is not None


def classify(query: str) -> QueryClass:
    """Classify a retrieval query by intent.

    Precedence matters: holding and validity keywords are checked before
    case references, because "what did the court hold in X" and "has X
    been overruled" are questions *about* X, not citation lookups.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query must be non-empty")

    if extract_opinion_id(query) is not None:
        return QueryClass.LOOKUP

    if extract_citation(query) is not None:
        return QueryClass.LOOKUP

    if _contains_validity_keyword(query):
        return QueryClass.VALIDITY

    if _contains_holding_keyword(query):
        return QueryClass.HOLDING

    if extract_case_reference(query) is not None:
        return QueryClass.LOOKUP

    return QueryClass.HOLDING


__all__ = [
    "QueryClass",
    "classify",
    "extract_citation",
    "extract_opinion_id",
    "extract_case_reference",
    "has_case_reference",
]
