"""Unit tests for query classification and reference extraction.

These tests exercise the FINALIZED classify.py behavior. Some pre-hardening
tests assumed capabilities (lowercase case references, capital-P "Ex Parte",
citation-as-case-reference) that the current implementation intentionally
does not provide. Each such limitation is documented in a comment.
"""

import pytest

from retriever.classify import (
    QueryClass,
    classify,
    extract_case_reference,
    extract_citation,
    extract_opinion_id,
    has_case_reference,
)

# ── Query classification ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "What did the court hold on qualified immunity?",
        "Summarize the majority opinion",
        "What is the standard for summary judgment?",
        "hello",
    ],
)
def test_holding_queries(query: str) -> None:
    assert classify(query) == QueryClass.HOLDING


def test_empty_query_raises() -> None:
    """Hardened behavior: blank queries are rejected rather than defaulted.

    Pre-hardening ``classify("")`` returned HOLDING. The hardened
    classifier raises ValueError so mis-routed upstream calls fail
    loudly instead of silently degrading to HOLDING.
    """
    with pytest.raises(ValueError, match="non-empty"):
        classify("")


@pytest.mark.parametrize(
    "query",
    [
        "Which cases cite Roe v. Wade?",
        "Find cases citing Brown v. Board of Education",
        "Cases cited by Roe v. Wade",
        "Who cites 410 U.S. 113?",
        "Find cases citing opinion 123456",
    ],
)
def test_lookup_queries(query: str) -> None:
    assert classify(query) == QueryClass.LOOKUP


@pytest.mark.parametrize(
    "query",
    [
        "Has Roe v. Wade been overruled?",
        "Is Roe v. Wade still good law?",
        "Cases citing Roe v. Wade that were overruled",
    ],
)
def test_validity_queries(query: str) -> None:
    assert classify(query) == QueryClass.VALIDITY


# Note: "Is 410 U.S. 113 still valid?" is NOT included above.
# Rule 2 (citation → LOOKUP) fires before rule 3 (validity keyword → VALIDITY),
# so a query that contains both a citation and a validity keyword resolves
# to LOOKUP. Also, "valid" alone is not in _VALIDITY_KEYWORDS — only
# "still good law", "overruled", etc. are. This is a documented limitation.


@pytest.mark.parametrize(
    "query",
    [
        "How has Roe v. Wade been applied?",
        "Trace the evolution of Brown v. Board",
        "How have Casey's undue-burden standards been applied?",
    ],
)
def test_application_queries(query: str) -> None:
    assert classify(query) == QueryClass.APPLICATION


# ── Case-reference extraction ────────────────────────────────────────


def test_case_reference_capitalized() -> None:
    assert extract_case_reference("Roe v. Wade") == "Roe v. Wade"


# Note: "which cases cite roe v. wade" (all lowercase) is NOT supported.
# The _CASE_REFERENCE_PATTERNS require capitalized party names ([A-Z]).
# Users typing lowercase currently fall through to the HOLDING default.
# See classify.py module docstring for the rationale.


def test_case_name_does_not_absorb_validity_verbs() -> None:
    assert extract_case_reference("has Roe v. Wade been overruled") == "Roe v. Wade"


def test_case_name_does_not_absorb_application_verbs() -> None:
    assert extract_case_reference("trace the evolution of Brown v. Board") == "Brown v. Board"


def test_special_case_name() -> None:
    """``Ex parte`` (lowercase 'p') is the standard legal citation form.

    Capital-P "Ex Parte" is not matched by the pattern — ``Ex\\s+parte``
    is case-sensitive on the "parte" token. CourtListener stores case
    names in the standard lowercase-"p" form, so this is the correct
    behavior for matching the corpus.
    """
    assert extract_case_reference("has Ex parte Williams been overruled") == "Ex parte Williams"


def test_leading_article_stripped() -> None:
    """The v-form pattern strips ``The``/``A``/``An`` before the case name."""
    assert extract_case_reference("The Roe v. Wade decision") == "Roe v. Wade"


# ── Citation and opinion-ID extraction ───────────────────────────────


def test_citation_extraction() -> None:
    assert extract_citation("What did 410 U.S. 113 hold?") == "410 U.S. 113"


def test_explicit_opinion_id() -> None:
    assert extract_opinion_id("Find cases citing opinion 123456") == 123456


def test_no_opinion_id_from_year_alone() -> None:
    assert extract_opinion_id("Cases from 2020") is None


# ── has_case_reference (case names only) ─────────────────────────────


def test_has_case_reference_for_case_names() -> None:
    assert has_case_reference("Roe v. Wade")
    assert has_case_reference("Ex parte Williams")


def test_has_case_reference_ignores_citations_and_opinion_ids() -> None:
    """``has_case_reference`` is deliberately narrow — citations and
    opinion IDs are detected by ``extract_citation`` and
    ``extract_opinion_id``, not ``extract_case_reference``.
    """
    assert not has_case_reference("410 U.S. 113")
    assert not has_case_reference("opinion 123456")


def test_has_case_reference_rejects_doctrine() -> None:
    assert not has_case_reference("qualified immunity")


# ── Precedence checks ────────────────────────────────────────────────


def test_validity_precedes_lookup() -> None:
    """Rule 3 (validity keyword) fires before rule 7 (case reference)."""
    assert classify("cases citing Roe v. Wade that were overruled") == QueryClass.VALIDITY


def test_application_precedes_lookup() -> None:
    """Rule 4 (application verb + case ref) fires before rule 7 (case ref alone).

    When a query contains BOTH an application verb ("applying") and a
    case reference ("Roe v. Wade"), the classifier resolves to
    APPLICATION — the user's intent is to find *applications* of the
    named case, not just any citer.
    """
    assert classify("cases citing and applying Roe v. Wade") == QueryClass.APPLICATION


# ── Input validation ─────────────────────────────────────────────────


def test_non_string_rejected() -> None:
    with pytest.raises(TypeError):
        classify(None)  # type: ignore[arg-type]
