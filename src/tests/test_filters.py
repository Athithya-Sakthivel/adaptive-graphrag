import pytest

from indexing_pipeline.filters import (
    is_noise_chunk,
    normalize_precedential_status,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Valid statuses — case-insensitive, whitespace-tolerant
        ("Published", "Published"),
        ("published", "Published"),
        ("  PUBLISHED  ", "Published"),
        ("Unpublished", "Unpublished"),
        ("In-chambers", "In-chambers"),
        ("In-Chambers", "In-chambers"),  # case-insensitive, hyphen preserved
        ("relating-to", "Relating-to"),
        ("Errata", "Errata"),
        # Garbage — must become None, never coerced
        ("2", None),
        ("5", None),
        (" Asst. Atty. Gen.", None),
        (None, None),
        ("", None),
        ("   ", None),
        (float("nan"), None),
        (float("inf"), None),
        (0, None),
        (True, None),
    ],
)
def test_normalize_precedential_status(raw, expected):
    assert normalize_precedential_status(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "* * *",
        "---",
        "...",
        "123",
        "2020",
        "42",
        "ALASKA",  # 6 chars, no lowercase, no digits
        "I.",  # 2 chars
        "II.",
        "U.S.",  # 4 chars — as a whole chunk body, noise
        "=" * 20,  # no alnum
    ],
)
def test_is_noise_chunk_rejects(text):
    assert is_noise_chunk(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "410 U.S. 113",
        "Roe v. Wade, 410 U.S. 113 (1973)",
        "See generally",
        "APPEAL FROM THE CIRCUIT COURT",
        "§ 1983",  # 7 chars, has digits → kept
        "REVERSED.",  # 9 chars, > 8 threshold → kept
        "Affirmed.",
        "Id.",  # has lowercase 'd'
        "supra",
        "United States",
    ],
)
def test_is_noise_chunk_keeps(text):
    assert is_noise_chunk(text) is False


def test_is_noise_chunk_rejects_non_string():
    with pytest.raises(TypeError):
        is_noise_chunk(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_noise_chunk(42)  # type: ignore[arg-type]


def test_is_noise_chunk_threshold_overridable():
    # "§§§§" — 4 alnum symbols, 0 letters. At ratio 0.0 it should pass
    # (alnum count is 0? no — § is not alnum). Let me pick a real case:
    # "a...." — 1 alnum, 4 non-alnum. ratio = 1/5 = 0.2
    text = "a...."
    assert is_noise_chunk(text, min_alnum_ratio=0.30) is True
    assert is_noise_chunk(text, min_alnum_ratio=0.10) is False
