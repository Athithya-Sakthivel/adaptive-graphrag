"""Unit tests for Filters metadata restrictions.

Filters is a frozen dataclass. This file exercises defaults, case-ID
normalization/intersection, date coercion, validation, and the frozen
guard.

Note on ``test_frozen``
-----------------------
A frozen dataclass raises ``FrozenInstanceError`` (a subclass of
``AttributeError``) on any attribute write. The test exercises that
guard through ``setattr()`` rather than a syntactic assignment
(``f.court = "scotus"``) because basedpyright reports
``reportAttributeAccessIssue`` on the syntactic form for frozen
dataclasses — even though it correctly raises at runtime. The
function-call form hits the same runtime path without tripping the
static checker.
"""

from datetime import date, datetime

import pytest

from retriever.filters import Filters

# ── Defaults ────────────────────────────────────────────────────────


def test_defaults_all_none() -> None:
    f = Filters()

    assert f.court is None
    assert f.jurisdiction is None
    assert f.date_from is None
    assert f.date_to is None
    assert f.case_ids is None
    assert f.published_only is False


# ── Case-ID handling ────────────────────────────────────────────────


def test_empty_case_ids_means_no_results() -> None:
    f = Filters(case_ids=[])

    assert f.case_ids == []
    assert f.matches_nothing


def test_case_ids_deduplicated() -> None:
    f = Filters(case_ids=[1, 2, 2, 3])

    assert f.case_ids == [1, 2, 3]


def test_case_ids_defensive_copy() -> None:
    f = Filters(case_ids=[1, 2])

    returned = f.case_ids
    assert returned is not None
    returned.append(999)

    # Mutating the returned list must not affect the stored filter.
    assert f.case_ids == [1, 2]


def test_boolean_is_not_valid_case_id() -> None:
    with pytest.raises(TypeError):
        Filters(case_ids=[True])


def test_intersect_case_ids() -> None:
    f = Filters(case_ids=[1, 2, 3])

    narrowed = f.with_case_ids([2, 3, 4])

    assert narrowed.case_ids == [2, 3]


def test_unrestricted_case_ids_accept_candidates() -> None:
    f = Filters()

    narrowed = f.with_case_ids([10, 20])

    assert narrowed.case_ids == [10, 20]


def test_empty_original_filter_remains_empty() -> None:
    f = Filters(case_ids=[])

    narrowed = f.with_case_ids([10, 20])

    assert narrowed.matches_nothing


# ── Frozen guard ────────────────────────────────────────────────────


def test_frozen() -> None:
    """Frozen dataclass refuses mutation at runtime.

    The syntactic assignment is intentional: it triggers the runtime
    guard. basedpyright correctly reports the write as read-only, but
    the warning is noise here — the test *wants* the write to fail.
    """
    f = Filters(court="ca9")

    with pytest.raises(AttributeError):
        f.court = "scotus"  # pyright: ignore[reportAttributeAccessIssue]


# ── Full configuration and date coercion ────────────────────────────


def test_full_configuration() -> None:
    f = Filters(
        court="ca9",
        jurisdiction="us-ca9",
        published_only=True,
        date_from=date(2020, 1, 1),
        date_to=date(2024, 12, 31),
        case_ids=[1, 2, 3],
    )

    assert f.published_only
    assert f.case_ids == [1, 2, 3]
    assert f.date_from == date(2020, 1, 1)
    assert f.date_to == date(2024, 12, 31)


def test_datetime_converted_to_date() -> None:
    f = Filters(date_from=datetime(2020, 1, 1, 12, 30))

    assert f.date_from == date(2020, 1, 1)


def test_invalid_date_range() -> None:
    with pytest.raises(ValueError):
        Filters(
            date_from=date(2025, 1, 1),
            date_to=date(2020, 1, 1),
        )


# ── SQL parameter marshaling ────────────────────────────────────────


def test_sql_parameters_are_arrays() -> None:
    f = Filters(case_ids=[1, 2])

    assert f.sql_params()["case_ids"] == [1, 2]
