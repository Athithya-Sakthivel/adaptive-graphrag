"""Immutable, backward-compatible metadata filters.

Applied inside the SQL WHERE clause of every retrieval arm to preserve
recall under restrictive predicates (post-filtering destroys recall
because HNSW scans a fixed number of candidates before filtering).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True, init=False)
class Filters:
    """Immutable metadata filters for retrieval queries.

    Args:
        court: Court slug (e.g., "ca9", "scotus").
        jurisdiction: Jurisdiction code (e.g., "us", "us-ca9").
        published_only: Restrict to published opinions.
        date_from: Inclusive start date.
        date_to: Inclusive end date.
        case_ids: Restrict to specific case IDs.
    """

    court: str | None
    jurisdiction: str | None
    published_only: bool
    date_from: date | None
    date_to: date | None
    _case_ids: tuple[int, ...] | None

    def __init__(
        self,
        court: str | None = None,
        jurisdiction: str | None = None,
        published_only: bool = False,
        date_from: date | datetime | None = None,
        date_to: date | datetime | None = None,
        case_ids: Iterable[int] | None = None,
    ) -> None:
        def normalize_text(value: str | None, name: str) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str or None")
            return value.strip() or None

        def normalize_date(value: date | datetime | None, name: str) -> date | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, date):
                return value
            raise TypeError(f"{name} must be date, datetime, or None")

        court = normalize_text(court, "court")
        jurisdiction = normalize_text(jurisdiction, "jurisdiction")

        if not isinstance(published_only, bool):
            raise TypeError("published_only must be bool")

        date_from = normalize_date(date_from, "date_from")
        date_to = normalize_date(date_to, "date_to")

        if date_from is not None and date_to is not None and date_from > date_to:
            raise ValueError("date_from cannot exceed date_to")

        normalized_ids: tuple[int, ...] | None
        if case_ids is None:
            normalized_ids = None
        else:
            if isinstance(case_ids, (str, bytes)):
                raise TypeError("case_ids must contain int IDs")
            seen: set[int] = set()
            values: list[int] = []
            for case_id in case_ids:
                if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
                    raise ValueError(f"case ID must be positive int: {case_id!r}")
                if case_id not in seen:
                    seen.add(case_id)
                    values.append(case_id)
            normalized_ids = tuple(values)

        for name, value in (
            ("court", court),
            ("jurisdiction", jurisdiction),
            ("published_only", published_only),
            ("date_from", date_from),
            ("date_to", date_to),
            ("_case_ids", normalized_ids),
        ):
            object.__setattr__(self, name, value)

    @property
    def case_ids(self) -> list[int] | None:
        """Defensive copy preserving documented list API."""
        return list(self._case_ids) if self._case_ids is not None else None

    @property
    def matches_nothing(self) -> bool:
        """True if filters restrict to an empty result set."""
        return self._case_ids == ()

    @property
    def has_restrictions(self) -> bool:
        """True if any filter predicate is active."""
        return any(
            (
                self.court is not None,
                self.jurisdiction is not None,
                self.published_only,
                self.date_from is not None,
                self.date_to is not None,
                self._case_ids is not None,
            )
        )

    def sql_params(self) -> dict:
        """Return dict suitable for psycopg parameter binding."""
        return {
            "court": self.court,
            "jurisdiction": self.jurisdiction,
            "published_only": self.published_only,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "case_ids": list(self._case_ids) if self._case_ids is not None else None,
        }

    def with_case_ids(self, case_ids: Iterable[int]) -> Filters:
        """Return new Filters intersecting case_ids with existing restrictions."""
        candidates = tuple(case_ids)
        if self._case_ids is not None:
            allowed = set(self._case_ids)
            candidates = tuple(cid for cid in candidates if cid in allowed)
        return Filters(
            court=self.court,
            jurisdiction=self.jurisdiction,
            published_only=self.published_only,
            date_from=self.date_from,
            date_to=self.date_to,
            case_ids=candidates,
        )


__all__ = ["Filters"]
