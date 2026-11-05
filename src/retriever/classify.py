"""Query classification and reference extraction.

Precedence in ``classify()`` (documented because it decides routing):

    1. Explicit opinion ID            → LOOKUP
    2. Explicit citation              → LOOKUP
    3. Validity keyword present       → VALIDITY
    4. Application verb + case ref    → APPLICATION
    5. Application verb + doctrine    → APPLICATION  (no case ref)
    6. Holding keyword present        → HOLDING
    7. Case ref present, no other hit → LOOKUP
    8. Default                        → HOLDING

Rule 3 fires before rule 7 because "has X been overruled" is a
validity question even though X is a case reference. Rule 6 fires
before rule 7 for the same reason: "what did the court hold on X"
is a holding question even though X is a case reference.

Rule 5 (doctrine-application fallback) is what this module adds
relative to the pre-hardening version. A query like "how has
qualified immunity been applied" has no case reference and no
citation; without rule 5 it would default to HOLDING and route to
dense+sparse. Rule 5 classifies it as APPLICATION so the router can
decide to route to dense+sparse **as an application query** (no
graph seed available) rather than silently mis-route it as HOLDING.

`extract_case_reference` deliberately returns ``None`` for
doctrine-only APPLICATION queries — the doctrine ("qualified
immunity") is not a case reference and must not be used to seed
graph traversal. The router's APPLICATION branch handles the
no-seed case by skipping the graph arm.

Case-reference pattern ordering
-------------------------------
The patterns are tried in this order:

    1. ``Ex parte X`` / ``Ex rel. X`` / ``In re X`` / ``Matter of X``
    2. ``X v. Y`` / ``X v Y`` / ``X vs. Y`` / ``X versus Y``

This matters for queries like "Ex parte Young v. United States",
where both patterns can match. The Ex parte form is tried first so
the extracted reference preserves the "Ex parte" prefix (the actual
styling of many cases), rather than truncating to the trailing
"v." form.

The v-form pattern:
    * requires a following ``v``/``v.``/``vs``/``vs.``/``versus``
      token — the standard Latin abbreviation "v." alone is a legal
      case-name form and is matched here. Pre-hardening versions
      only accepted "vs"/"vs."/"versus", which silently failed to
      extract references in "Roe v. Wade".
    * strips a leading ``The``/``A``/``An`` before the case name so
      ``"The Roe v. Wade decision"`` extracts ``"Roe v. Wade"`` and
      not ``"The Roe v. Wade"``.

Known heuristic limitations
---------------------------
1. Rule 5 only inspects the phrase *before* the application verb,
   per design. Queries in the form "cases applying qualified
   immunity" (doctrine *after* the verb) fall through to the default
   HOLDING class. Extending to phrase-after-verb would help but
   risks false positives on queries like "what did the court hold
   on the application of X", where "application" is a topic noun,
   not a verb. Deferred.
2. "since" is included in the application-verb set because the
   hardening spec lists it, but it is by far the noisiest member —
   "how has the law changed since 2000" classifies as APPLICATION.
   The behaviour is defensible (the query *is* about temporal
   evolution of a doctrine) but the classifier is heuristic, not
   deterministic. If eval shows false positives on "since", remove
   it from ``_APPLICATION_VERB_PATTERN`` and the doctrine fallback
   stops matching such queries.
3. Case names containing lowercase connectives ("Board of
   Education v. Brown", "United States of America v. X") are
   captured only up to the first lowercase word ("Education v.
   Brown", "United States v. X"). The trigram fallback in
   ``exact.py`` resolves these against the full name because the
   overlap is high. Adding connective support to the pattern is a
   possible future improvement but is not required for correctness
   given the trigram safety net.
"""

from __future__ import annotations

import re
from enum import StrEnum


class QueryClass(StrEnum):
    HOLDING = "HOLDING"
    LOOKUP = "LOOKUP"
    VALIDITY = "VALIDITY"
    APPLICATION = "APPLICATION"


# ── Patterns ──────────────────────────────────────────────────────────

# Reporter citation: volume (digits, optional 1-letter suffix like "2d"
# or "3d"), reporter token (starts uppercase, may contain digits and
# periods like "F.3d", "U.S.", "F. Supp. 2d"), optional sub-series
# tokens, then a page number. Deliberately permissive — a false
# positive here routes to LOOKUP with the "citation" string never
# actually being valid, but ``resolve_case_ids`` in ``exact.py`` is the
# second line of defence. A tighter pattern would miss legitimate
# variants like "1 F. Supp. 2d 100" or "12 S. Ct. 345".
_CITATION_PATTERN = re.compile(
    r"\b\d+[a-z]?\s+"
    r"[A-Z][A-Za-z0-9.]{1,9}"
    r"(?:\s+[A-Z0-9][A-Za-z0-9.]{0,9}){0,2}"
    r"\s+\d+\b",
    re.UNICODE,
)

# "opinion 12345", "opinion id 12345", "opinion_id:12345"
_OPINION_ID_PATTERN = re.compile(
    r"\bopinion(?:_?id)?\s*[:#]?\s*(\d{1,10})\b",
    re.IGNORECASE,
)

# Case-name reference patterns. Each pattern's group(1) is the full
# matched name. Names are captured conservatively — the trailing word
# sequence allows up to 4 continuation words for names like
# "Board of Education v. Brown" but not so greedy that it swallows the
# rest of the sentence.
#
# Order matters: Ex parte forms are tried first so that queries like
# "Ex parte Young v. United States" capture the Ex parte styling
# rather than the trailing v. form.
_CASE_REFERENCE_PATTERNS = (
    # "Ex parte X" / "Ex rel. X" / "In re X" / "Matter of X"
    re.compile(
        r"\b((?:Ex\s+parte|Ex\s+rel\.?|In\s+re|Matter\s+of)\s+"
        r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,4})",
        re.UNICODE,
    ),
    # "X v. Y" / "X v Y" / "X vs. Y" / "X versus Y"
    # The v-token accepts: "v", "v.", "vs", "vs.", "versus".
    # The leading article strip prevents "The"/"A"/"An" from being
    # captured as part of the case name.
    re.compile(
        r"\b(?:The\s+|A\s+|An\s+)?"
        r"([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,4}"
        r"\s+v(?:\.|s\.?|ersus)?\s+"
        r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,4})",
        re.UNICODE,
    ),
)

# Validity keywords (substring match; overlapping stems such as
# "overrule"/"overruled"/"overruling" are all listed explicitly so
# substring matching is unambiguous).
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

# Application verbs. Word-boundary anchored so "misapplied" and
# "reapplied" do not match.
_APPLICATION_VERB_PATTERN = re.compile(
    r"\b(applied|applying|application|applies|apply"
    r"|evolution|trace[sd]?|tracing"
    r"|since)\b",
    re.IGNORECASE,
)

# Holding phrases (substring match) and holding words (word-boundary
# match). Multi-word phrases are checked as substrings because their
# internal structure is already specific; single words go through the
# boundary check to avoid matching "withholding", "shareholder", etc.
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

_HOLDING_WORDS_PATTERN = re.compile(
    r"\b(holding|held|holds)\b",
    re.IGNORECASE,
)


# ── Extraction ────────────────────────────────────────────────────────


def extract_citation(query: str) -> str | None:
    """Return the first reporter citation in ``query``, or ``None``."""
    match = _CITATION_PATTERN.search(query)
    if match is None:
        return None
    return match.group(0).strip()


def extract_opinion_id(query: str) -> int | None:
    """Return the first explicit opinion ID in ``query``, or ``None``."""
    match = _OPINION_ID_PATTERN.search(query)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_case_reference(query: str) -> str | None:
    """Return the first case-name reference in ``query``, or ``None``.

    Doctrine phrases are deliberately *not* returned. The APPLICATION
    doctrine fallback in ``classify()`` uses a separate heuristic and
    leaves this function's contract unchanged.

    A leading ``The``/``A``/``An`` is stripped from the v-form
    reference so that ``"The Roe v. Wade decision"`` yields
    ``"Roe v. Wade"``, not ``"The Roe v. Wade"``.
    """
    for pattern in _CASE_REFERENCE_PATTERNS:
        match = pattern.search(query)
        if match is not None:
            return match.group(1).strip()
    return None


def has_case_reference(query: str) -> bool:
    """Return True when ``query`` contains an extractable case reference."""
    return extract_case_reference(query) is not None


# ── Doctrine-phrase heuristic ────────────────────────────────────────

# Question-word prefixes stripped before testing the phrase-before-verb
# as a doctrine candidate.
_LEADING_PHRASES = (
    "in what ways has ",
    "in what ways have ",
    "in what way has ",
    "in what way have ",
    "how has ",
    "how have ",
    "how does ",
    "how did ",
    "how is ",
    "how was ",
    "how were ",
    "what has ",
    "what have ",
    "what does ",
    "what did ",
    "when has ",
    "when was ",
    "when did ",
    "where has ",
    "where was ",
    "where did ",
)

# Auxiliary trailing tokens stripped from the end of the before-verb
# phrase, because "qualified immunity been applied" has the noun phrase
# before "been", not before "applied".
_TRAILING_AUX = (
    " has been",
    " have been",
    " had been",
    " been",
    " was",
    " were",
    " is",
    " are",
    " has",
    " have",
    " had",
)

_LEADING_ARTICLES = ("the ", "a ", "an ")

# Markers indicating a case name, not a doctrine. These should already
# be caught by ``extract_case_reference``, but ``_looks_like_doctrine``
# runs even when no reference was extracted, so it stays defensive.
_CASE_MARKER_PATTERN = re.compile(
    r"\s(?:v\.?|vs\.?|versus)\s",
    re.IGNORECASE,
)


def _extract_doctrine_phrase(query: str, verb_start: int) -> str | None:
    """Return the phrase before ``verb_start``, stripped of scaffolding.

    Returns ``None`` when nothing usable remains (e.g. the verb is the
    first word, or the prefix was entirely scaffolding). The caller is
    responsible for deciding whether the returned phrase actually looks
    like a doctrine.
    """
    before = query[:verb_start].strip()
    if not before:
        return None

    lowered = before.lower()

    for lead in _LEADING_PHRASES:
        if lowered.startswith(lead):
            before = before[len(lead) :]
            lowered = before.lower()
            break

    # Repeat stripping — a phrase like "the qualified immunity has been"
    # needs both "has been" and "the" removed.
    for _ in range(2):
        stripped_any = False
        for trailing in _TRAILING_AUX:
            if lowered.endswith(trailing):
                before = before[: -len(trailing)].rstrip()
                lowered = before.lower()
                stripped_any = True
                break
        if not stripped_any:
            break

    before = before.strip(" ?.,;:'\"")
    if not before:
        return None

    low = before.lower()
    for article in _LEADING_ARTICLES:
        if low.startswith(article):
            before = before[len(article) :]
            low = before.lower()
            break

    before = before.strip()
    return before or None


def _looks_like_doctrine(phrase: str) -> bool:
    """Heuristic: 2+ words, no case-name markers, no leading case form."""
    words = phrase.split()
    if len(words) < 2:
        return False

    if _CASE_MARKER_PATTERN.search(phrase):
        return False

    low = phrase.lower()
    for marker in ("ex parte", "ex rel", "in re", "matter of"):
        if low.startswith(marker):
            return False

    return True


# ── Classification ────────────────────────────────────────────────────


def _contains_validity_keyword(query: str) -> bool:
    lower = query.lower()
    return any(keyword in lower for keyword in _VALIDITY_KEYWORDS)


def _contains_holding_keyword(query: str) -> bool:
    lower = query.lower()
    for phrase in _HOLDING_PHRASES:
        if phrase in lower:
            return True
    return _HOLDING_WORDS_PATTERN.search(lower) is not None


def _application_verb_match(query: str) -> re.Match[str] | None:
    """Return the first application-verb match, or ``None``.

    Leftmost match wins, so "how has X been applied since 2000" returns
    the "applied" match (which has a usable before-phrase) rather than
    the "since" match (which does not).
    """
    return _APPLICATION_VERB_PATTERN.search(query)


def classify(query: str) -> QueryClass:
    """Classify a retrieval query. See module docstring for precedence."""
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query must be non-empty")

    # Rule 1: explicit opinion ID
    if extract_opinion_id(query) is not None:
        return QueryClass.LOOKUP

    # Rule 2: explicit citation
    if extract_citation(query) is not None:
        return QueryClass.LOOKUP

    # Rule 3: validity keyword
    if _contains_validity_keyword(query):
        return QueryClass.VALIDITY

    case_reference = extract_case_reference(query)
    verb_match = _application_verb_match(query)

    # Rule 4: application verb + case reference
    if verb_match is not None and case_reference is not None:
        return QueryClass.APPLICATION

    # Rule 5: application verb + doctrine phrase (no case reference)
    if verb_match is not None and case_reference is None:
        phrase = _extract_doctrine_phrase(query, verb_match.start())
        if phrase is not None and _looks_like_doctrine(phrase):
            return QueryClass.APPLICATION

    # Rule 6: holding keyword
    if _contains_holding_keyword(query):
        return QueryClass.HOLDING

    # Rule 7: case reference alone
    if case_reference is not None:
        return QueryClass.LOOKUP

    # Rule 8: default
    return QueryClass.HOLDING


__all__ = [
    "QueryClass",
    "classify",
    "extract_citation",
    "extract_opinion_id",
    "extract_case_reference",
    "has_case_reference",
]
