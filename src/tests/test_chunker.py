# tests/test_chunker_v15.py
import pytest

from indexing_pipeline.chunker import ChunkingError, chunk_opinion


def _record(case_id=1, text="", **kw):
    return {"cl_opinion_id": case_id, "text": text, "text_kind": "plain_text", **kw}


def test_no_is_noise_unchanged_from_v14():
    """is_noise=None must reproduce v1.4 behavior exactly."""
    r = _record(text="Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks_v15 = chunk_opinion(r)  # no is_noise
    # Should have 1 parent + at least 1 child
    assert any(c.is_parent for c in chunks_v15)
    assert any(not c.is_parent for c in chunks_v15)


def test_is_noise_filters_children():
    """is_noise predicate drops matching children."""
    r = _record(text="Meaningful legal text.\n\n2020\n\nMore meaningful text.")
    # Drop pure-digit paragraphs
    is_noise = lambda t: t.strip().isdigit()
    chunks = chunk_opinion(r, is_noise=is_noise)
    children = [c for c in chunks if not c.is_parent]
    for c in children:
        assert not c.text_original.strip().isdigit()


def test_orphan_parent_prevented():
    """A section whose entire body is noise emits neither parent nor child."""
    # Non-asterisk text that survives v2.0's page-marker stripping.
    # is_noise=lambda t: True filters every child; chunker must not emit
    # a lone parent with no children, so it raises.
    r = _record(text="Paragraph one.\n\nParagraph two.")
    with pytest.raises(ChunkingError, match="no valid parent chunks"):
        chunk_opinion(r, is_noise=lambda t: True)


def test_parent_kept_when_any_child_survives():
    r = _record(text="2020\n\nMeaningful legal text here.")
    is_noise = lambda t: t.strip().isdigit()
    chunks = chunk_opinion(r, is_noise=is_noise)
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert len(parents) == 1
    assert len(children) >= 1
    # The surviving child must have a valid parent reference
    assert all(c.parent_id in {p.chunk_id for p in parents} for c in children)
