"""Immutable, backward-compatible metadata filters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True, init=False)
class Filters:
    court: str | None
    jurisdiction: str | None
    published_only: bool
    date_from: date | None
    date_to: date | None

    # Internal immutable representation. The public case_ids property
    # returns a defensive list copy to preserve the documented API.
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
                raise TypeError(f"{name} must be string or None")
            return value.strip() or None

        def normalize_date(
            value: date | datetime | None,
            name: str,
        ) -> date | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, date):
                return value
            raise TypeError(f"{name} must be date, datetime or None")

        court = normalize_text(court, "court")
        jurisdiction = normalize_text(jurisdiction, "jurisdiction")

        if not isinstance(published_only, bool):
            raise TypeError("published_only must be bool")

        date_from = normalize_date(date_from, "date_from")
        date_to = normalize_date(date_to, "date_to")

        if date_from is not None and date_to is not None:
            if date_from > date_to:
                raise ValueError("date_from cannot exceed date_to")

        normalized_ids: tuple[int, ...] | None

        if case_ids is None:
            normalized_ids = None
        else:
            if isinstance(case_ids, (str, bytes)):
                raise TypeError("case_ids must contain integer IDs")

            seen: set[int] = set()
            values: list[int] = []

            for case_id in case_ids:
                if isinstance(case_id, bool) or not isinstance(case_id, int):
                    raise TypeError(f"case ID must be integer: {case_id!r}")
                if case_id <= 0:
                    raise ValueError("case IDs must be positive")

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
        """Return a defensive copy, preserving the documented list API."""

        if self._case_ids is None:
            return None
        return list(self._case_ids)

    @property
    def matches_nothing(self) -> bool:
        return self._case_ids == ()

    @property
    def has_restrictions(self) -> bool:
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
        return {
            "court": self.court,
            "jurisdiction": self.jurisdiction,
            "published_only": self.published_only,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "case_ids": (list(self._case_ids) if self._case_ids is not None else None),
        }

    def with_case_ids(self, case_ids: Iterable[int]) -> Filters:
        """Intersect additional case candidates with existing restrictions."""

        candidates = tuple(case_ids)

        if self._case_ids is not None:
            allowed = set(self._case_ids)
            candidates = tuple(case_id for case_id in candidates if case_id in allowed)

        return Filters(
            court=self.court,
            jurisdiction=self.jurisdiction,
            published_only=self.published_only,
            date_from=self.date_from,
            date_to=self.date_to,
            case_ids=candidates,
        )


__all__ = ["Filters"]
