"""Unit tests for Filters metadata restrictions.

Filters is a frozen dataclass. This file exercises defaults, case-ID
normalization/intersection, date coercion, validation, and the frozen
guard.

Note on test_frozen
-------------------
A frozen dataclass raises FrozenInstanceError (a subclass of
AttributeError) on any attribute write. The syntactic assignment is
intentional: it triggers the runtime guard. basedpyright correctly
reports the write as read-only, but the warning is noise here — the
test wants the write to fail.
"""

from __future__ import annotations

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
    assert not f.has_restrictions
    assert not f.matches_nothing


# ── Case-ID handling ────────────────────────────────────────────────


def test_empty_case_ids_means_no_results() -> None:
    f = Filters(case_ids=[])

    assert f.case_ids == []
    assert f.matches_nothing


def test_case_ids_deduplicated() -> None:
    f = Filters(case_ids=[1, 2, 2, 3])

    assert f.case_ids == [1, 2, 3]


def test_case_ids_order_preserved() -> None:
    f = Filters(case_ids=[3, 1, 2])

    assert f.case_ids == [3, 1, 2]


def test_case_ids_defensive_copy() -> None:
    f = Filters(case_ids=[1, 2])

    returned = f.case_ids
    assert returned is not None
    returned.append(999)

    assert f.case_ids == [1, 2]


def test_boolean_is_not_valid_case_id() -> None:
    """bool subclasses int, so True would silently become case_id 1.

    Filters must reject it explicitly. ValueError matches the file's
    convention for invalid case IDs.
    """
    with pytest.raises(ValueError, match="positive int"):
        Filters(case_ids=[True])


def test_false_is_not_valid_case_id() -> None:
    with pytest.raises(ValueError, match="positive int"):
        Filters(case_ids=[False])


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_case_id_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="positive int"):
        Filters(case_ids=[bad])


def test_non_int_case_id_rejected() -> None:
    with pytest.raises(ValueError, match="positive int"):
        Filters(case_ids=["1"])  # type: ignore[list-item]


def test_string_case_ids_container_rejected() -> None:
    with pytest.raises(TypeError):
        Filters(case_ids="123")  # type: ignore[arg-type]


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


def test_disjoint_intersection_is_empty() -> None:
    f = Filters(case_ids=[1, 2])

    narrowed = f.with_case_ids([8, 9])

    assert narrowed.matches_nothing


def test_intersection_preserves_other_restrictions() -> None:
    f = Filters(court="ca9", published_only=True, case_ids=[1, 2, 3])

    narrowed = f.with_case_ids([2])

    assert narrowed.court == "ca9"
    assert narrowed.published_only
    assert narrowed.case_ids == [2]


# ── Frozen guard ────────────────────────────────────────────────────


def test_frozen() -> None:
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
    assert f.has_restrictions


def test_datetime_converted_to_date() -> None:
    f = Filters(date_from=datetime(2020, 1, 1, 12, 30))

    assert f.date_from == date(2020, 1, 1)


def test_invalid_date_range() -> None:
    with pytest.raises(ValueError):
        Filters(date_from=date(2025, 1, 1), date_to=date(2020, 1, 1))


def test_equal_dates_allowed() -> None:
    f = Filters(date_from=date(2020, 1, 1), date_to=date(2020, 1, 1))

    assert f.date_from == f.date_to


def test_whitespace_court_normalized_to_none() -> None:
    assert Filters(court="   ").court is None


def test_court_stripped() -> None:
    assert Filters(court="  ca9  ").court == "ca9"


def test_non_bool_published_only_rejected() -> None:
    with pytest.raises(TypeError):
        Filters(published_only="yes")  # type: ignore[arg-type]


# ── SQL parameter marshaling ────────────────────────────────────────


def test_sql_parameters_are_arrays() -> None:
    f = Filters(case_ids=[1, 2])

    assert f.sql_params()["case_ids"] == [1, 2]


def test_sql_params_unrestricted_are_none() -> None:
    params = Filters().sql_params()

    assert params["court"] is None
    assert params["jurisdiction"] is None
    assert params["case_ids"] is None
    assert params["published_only"] is False


def test_sql_params_empty_case_ids_is_empty_list() -> None:
    """Empty list (not None) so SQL `= ANY('{}')` matches nothing."""
    assert Filters(case_ids=[]).sql_params()["case_ids"] == []
