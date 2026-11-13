"""Unit tests for query classification and reference extraction.

These tests exercise the finalized classify.py: three query classes
(HOLDING, LOOKUP, VALIDITY), template stripping in extraction, and
holding-keyword precedence over case references.

Known limitations are asserted explicitly rather than skipped, so a
future change in behavior surfaces as a test failure.
"""

from __future__ import annotations

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


@pytest.mark.parametrize(
    "query",
    [
        "We hold that the accomplice's testimony was sufficiently corroborated.",
        "The court held that the statute was unconstitutional.",
        "This court holds that the motion must be denied.",
        "The holding of the lower court was affirmed.",
    ],
)
def test_holding_keyword_variants(query: str) -> None:
    """All four inflections (hold/holding/held/holds) route to HOLDING."""
    assert classify(query) == QueryClass.HOLDING


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        classify("")


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_whitespace_only_query_raises(query: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        classify(query)


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
        "Was Miranda abrogated by a later decision?",
    ],
)
def test_validity_queries(query: str) -> None:
    assert classify(query) == QueryClass.VALIDITY


def test_only_three_query_classes_exist() -> None:
    """APPLICATION and multi-hop traversal were removed from the system."""
    assert {c.value for c in QueryClass} == {"HOLDING", "LOOKUP", "VALIDITY"}


# ── Case-reference extraction ────────────────────────────────────────


def test_case_reference_capitalized() -> None:
    assert extract_case_reference("Roe v. Wade") == "Roe v. Wade"


def test_lookup_template_prefix_stripped() -> None:
    """The greedy regex captures the template; extraction must strip it."""
    assert extract_case_reference("Which cases cite Delevie v. State?") == "Delevie v. State"


def test_validity_template_prefix_and_suffix_stripped() -> None:
    assert extract_case_reference("Has Davis v. State been overruled?") == "Davis v. State"


def test_find_cases_citing_template_stripped() -> None:
    assert extract_case_reference("Find cases citing Swain v. Terry") == "Swain v. Terry"


def test_lowercase_case_reference_not_supported() -> None:
    """Party names must be capitalized; all-lowercase input yields None.

    Users typing lowercase fall through to the HOLDING default. This is
    a documented limitation of the [A-Z]-anchored patterns.
    """
    assert extract_case_reference("which cases cite roe v. wade") is None


def test_case_name_does_not_absorb_validity_verbs() -> None:
    assert extract_case_reference("has Roe v. Wade been overruled") == "Roe v. Wade"


def test_case_name_does_not_absorb_trailing_prose() -> None:
    assert extract_case_reference("trace the evolution of Brown v. Board") == "Brown v. Board"


@pytest.mark.parametrize("name", ["Ex parte Williams", "Ex Parte Williams"])
def test_special_case_name_both_capitalizations(name: str) -> None:
    """Both 'Ex parte' and 'Ex Parte' are matched via [Pp]arte."""
    assert extract_case_reference(f"has {name} been overruled") == name


def test_in_re_pattern() -> None:
    assert extract_case_reference("In re Marriage of Smith") == "In re Marriage of Smith"


def test_leading_article_not_stripped() -> None:
    """Known limitation: leading articles and trailing nouns are absorbed.

    The v-form pattern has no article-stripping group, so 'The' and
    'decision' are captured as part of the name. Exact resolution will
    miss; the trigram fallback in exact.resolve_case_ids is the recovery
    path for this shape.
    """
    assert extract_case_reference("The Roe v. Wade decision") == "The Roe v. Wade decision"


# ── Citation and opinion-ID extraction ───────────────────────────────


def test_citation_extraction() -> None:
    assert extract_citation("What did 410 U.S. 113 hold?") == "410 U.S. 113"


def test_no_citation_in_plain_prose() -> None:
    assert extract_citation("qualified immunity standard") is None


def test_explicit_opinion_id() -> None:
    assert extract_opinion_id("Find cases citing opinion 123456") == 123456


def test_opinion_id_with_colon() -> None:
    assert extract_opinion_id("opinion_id: 42") == 42


def test_no_opinion_id_from_year_alone() -> None:
    assert extract_opinion_id("Cases from 2020") is None


# ── has_case_reference (case names only) ─────────────────────────────


def test_has_case_reference_for_case_names() -> None:
    assert has_case_reference("Roe v. Wade")
    assert has_case_reference("Ex parte Williams")


def test_has_case_reference_ignores_citations_and_opinion_ids() -> None:
    """Narrow by design: citations and IDs have their own extractors."""
    assert not has_case_reference("410 U.S. 113")
    assert not has_case_reference("opinion 123456")


def test_has_case_reference_rejects_doctrine() -> None:
    assert not has_case_reference("qualified immunity")


# ── Precedence checks ────────────────────────────────────────────────


def test_opinion_id_precedes_everything() -> None:
    assert classify("opinion 123456 overruled holding") == QueryClass.LOOKUP


def test_citation_precedes_validity_keyword() -> None:
    """Rule 2 (citation) fires before rule 3 (validity keyword).

    'Is 410 U.S. 113 still good law?' resolves to LOOKUP, not VALIDITY.
    Documented limitation: citation presence wins over validity intent.
    """
    assert classify("Is 410 U.S. 113 still good law?") == QueryClass.LOOKUP


def test_validity_precedes_case_reference() -> None:
    assert classify("cases citing Roe v. Wade that were overruled") == QueryClass.VALIDITY


def test_holding_precedes_case_reference() -> None:
    """Rule 4 (holding keyword) fires before rule 5 (case reference).

    'What did the court hold in X' is a question about X, not a request
    for the list of cases citing X.
    """
    assert classify("What did the court hold in Roe v. Wade?") == QueryClass.HOLDING


def test_holding_text_citing_precedent_misroutes_to_lookup() -> None:
    """Known limitation: passage-shaped queries with no holding keyword.

    A holding excerpt that cites other precedent but contains no
    hold/holding/held/holds token falls through to rule 5 and routes to
    LOOKUP. Sparse retrieval still runs in that path, which is why
    measured HOLDING recall remains high despite these misroutes.
    """
    query = "We conclude that the evidence was clear. Smith v. Jones controls the outcome."
    assert classify(query) == QueryClass.LOOKUP


# ── Input validation ─────────────────────────────────────────────────


def test_non_string_rejected() -> None:
    with pytest.raises(TypeError):
        classify(None)  # type: ignore[arg-type]
