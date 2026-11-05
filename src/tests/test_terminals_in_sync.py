"""Assert that the three places defining "chunk ends in punctuation" agree.

The terminal character set is unavoidably duplicated:
  - chunker._SENTENCE_END_RE  → drives _trim_to_sentence_end (Python)
  - db.VALIDATION_SQL gate_3  → drives validate gate (SQL)
  - inspect_chunks.END_PUNCT_RE → drives pre-load inspect (Python)

If any of the three diverges, gate_3 becomes untrustworthy: the chunker
can produce chunks that pass one check and fail the other. This test fails
the moment they drift, so the class of bug is caught in CI, not in a run.
"""

from __future__ import annotations

import re

from indexing_pipeline import chunker, db
from utils.inspect_chunks import END_PUNCT_RE


def _extract_char_class(pattern: re.Pattern[str]) -> frozenset[str]:
    """Return the set of characters that pattern's first bracket class accepts."""
    src = pattern.pattern
    # Find the first [...] bracket class
    match = re.search(r"\[([^\]]+)\]", src)
    assert match, f"no bracket class in {src!r}"
    body = match.group(1)
    # Expand simple escape sequences we care about
    body = body.replace("\\]", "]")
    return frozenset(body)


def _extract_from_sql(sql: str) -> frozenset[str]:
    """Return the terminal chars from gate_3's SQL regex."""
    # gate_3 has: rtrim(text_original) ~ '[.!?…:)"”’\\]]$'
    match = re.search(r"rtrim\(text_original\) ~ '\[([^\]]+)\]", sql)
    assert match, f"gate_3 SQL structure changed: {sql[:100]!r}"
    body = match.group(1).replace("\\\\", "\\").replace("\\]", "]")
    return frozenset(body)


def test_all_three_terminal_sets_agree() -> None:
    chunker_chars = _extract_char_class(chunker._SENTENCE_END_RE)
    inspect_chars = _extract_char_class(END_PUNCT_RE)
    sql_chars = _extract_from_sql(db.VALIDATION_SQL["gate_3_boundary_integrity"])

    assert chunker_chars == inspect_chars, (
        f"chunker._SENTENCE_END_RE and inspect_chunks.END_PUNCT_RE disagree:\n"
        f"  only in chunker:  {chunker_chars - inspect_chars}\n"
        f"  only in inspect:  {inspect_chars - chunker_chars}"
    )
    assert chunker_chars == sql_chars, (
        f"chunker._SENTENCE_END_RE and db gate_3 disagree:\n"
        f"  only in chunker:  {chunker_chars - sql_chars}\n"
        f"  only in db:       {sql_chars - chunker_chars}"
    )


def test_terminal_set_contains_documented_chars() -> None:
    """Lock down the specific chars so a regression is caught even if
    all three files drift together."""
    expected = frozenset(".!?…:;")
    actual = _extract_char_class(chunker._SENTENCE_END_RE)
    assert actual == expected, (
        f"terminal set changed: {sorted(actual)} != {sorted(expected)}. "
        f"If intentional, update this test AND all three files together."
    )
