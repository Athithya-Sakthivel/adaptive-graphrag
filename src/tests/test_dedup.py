from pathlib import Path

import pytest

from indexing_pipeline.dedup import (
    DedupError,
    dedup_chunks,
)
from indexing_pipeline.io_utils import atomic_write_jsonl, read_jsonl


def _write_case(
    chunked_dir: Path, case_id: int, children: list[dict], parents: list[dict] | None = None
):
    """Helper: write a case file and its manifest entry."""
    rows = (parents or []) + children
    path = chunked_dir / f"{case_id}.jsonl"
    atomic_write_jsonl(path, rows)
    # Append to manifest
    manifest_path = chunked_dir / "_manifest.jsonl"
    existing = list(read_jsonl(manifest_path)) if manifest_path.is_file() else []
    existing.append(
        {
            "case_id": case_id,
            "chunk_count": len(rows),
            "embedding_count": len(children),
            "file_sha256": "sha256:test",
        }
    )
    atomic_write_jsonl(manifest_path, existing)


def _child(chunk_id: str, hash_suffix: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "is_parent": False,
        "text_hash": f"sha256:{hash_suffix:0>64}",
        "text_original": f"body of {chunk_id}",
        "parent_id": "parent",
    }


def _parent(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "is_parent": True,
        "text_hash": f"sha256:p_{chunk_id:0>58}",
        "text_original": f"parent {chunk_id}",
        "parent_id": None,
    }


def test_empty_dir_returns_empty_report(tmp_path):
    (tmp_path / "_manifest.jsonl").write_text("")
    report = dedup_chunks(tmp_path)
    assert report.cases_scanned == 0
    assert report.chunks_dropped == 0
    assert report.duplicate_hashes == 0


def test_missing_dir_raises(tmp_path):
    with pytest.raises(DedupError, match="does not exist"):
        dedup_chunks(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "1.jsonl").write_text("")
    with pytest.raises(DedupError, match="missing manifest"):
        dedup_chunks(tmp_path)


def test_manifest_file_mismatch_raises(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    # Delete the file but leave the manifest entry
    (tmp_path / "1.jsonl").unlink()
    with pytest.raises(DedupError, match="manifest/file mismatch"):
        dedup_chunks(tmp_path)


def test_no_duplicates_no_rewrite(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    _write_case(tmp_path, 2, [_child("2:c:0", "bbb")])
    report = dedup_chunks(tmp_path)
    assert report.duplicate_hashes == 0
    assert report.cases_rewritten == 0
    assert report.chunks_dropped == 0


def test_cross_case_duplicate_dropped_from_higher_case(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    _write_case(tmp_path, 2, [_child("2:c:0", "aaa")])
    report = dedup_chunks(tmp_path)
    assert report.duplicate_hashes == 1
    assert report.chunks_dropped == 1
    assert report.cases_rewritten == 1

    case1 = [r for r in read_jsonl(tmp_path / "1.jsonl") if not r["is_parent"]]
    case2 = [r for r in read_jsonl(tmp_path / "2.jsonl") if not r["is_parent"]]
    assert len(case1) == 1  # keeper (lowest case_id)
    assert len(case2) == 0  # dropped


def test_dedup_is_idempotent(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    _write_case(tmp_path, 2, [_child("2:c:0", "aaa")])
    r1 = dedup_chunks(tmp_path)
    r2 = dedup_chunks(tmp_path)
    assert r1.chunks_dropped == 1
    assert r2.chunks_dropped == 0
    assert r2.cases_rewritten == 0


def test_parents_never_dropped(tmp_path):
    _write_case(
        tmp_path,
        1,
        children=[_child("1:c:0", "aaa")],
        parents=[_parent("1:p:0")],
    )
    _write_case(
        tmp_path,
        2,
        children=[_child("2:c:0", "aaa")],
        parents=[_parent("2:p:0")],
    )
    dedup_chunks(tmp_path)
    case2 = list(read_jsonl(tmp_path / "2.jsonl"))
    parents_2 = [r for r in case2 if r["is_parent"]]
    assert len(parents_2) == 1  # parent retained


def test_manifest_updated_after_dedup(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    _write_case(tmp_path, 2, [_child("2:c:0", "aaa")])
    dedup_chunks(tmp_path)
    manifest = {r["case_id"]: r for r in read_jsonl(tmp_path / "_manifest.jsonl")}
    assert manifest[2]["embedding_count"] == 0
    assert "deduped_at" in manifest[2]


def test_intra_case_duplicate_warned_not_repaired(tmp_path):
    _write_case(
        tmp_path,
        1,
        [
            _child("1:c:0", "aaa"),
            _child("1:c:1", "aaa"),  # same hash, same case
        ],
    )
    report = dedup_chunks(tmp_path)
    assert report.chunks_dropped == 0
    assert len(report.intra_case_duplicates) == 1
    assert report.intra_case_duplicates[0]["case_id"] == 1
    # Both children still present — dedup doesn't repair
    case1 = [r for r in read_jsonl(tmp_path / "1.jsonl") if not r["is_parent"]]
    assert len(case1) == 2


def test_dry_run_does_not_modify(tmp_path):
    _write_case(tmp_path, 1, [_child("1:c:0", "aaa")])
    _write_case(tmp_path, 2, [_child("2:c:0", "aaa")])
    before_1 = (tmp_path / "1.jsonl").read_bytes()
    before_2 = (tmp_path / "2.jsonl").read_bytes()
    report = dedup_chunks(tmp_path, dry_run=True)
    assert report.chunks_dropped == 1
    assert (tmp_path / "1.jsonl").read_bytes() == before_1
    assert (tmp_path / "2.jsonl").read_bytes() == before_2
    assert not (tmp_path / "_dedup_report.json").exists()


def test_missing_text_hash_raises(tmp_path):
    _write_case(
        tmp_path,
        1,
        [
            {"chunk_id": "1:c:0", "is_parent": False, "text_original": "x"},
        ],
    )
    with pytest.raises(DedupError, match="missing valid text_hash"):
        dedup_chunks(tmp_path)
