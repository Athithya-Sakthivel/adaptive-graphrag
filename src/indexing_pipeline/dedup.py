"""Cross-case paragraph deduplication.

Post-chunk, pre-embed. Exact-hash only. Stdlib only. Idempotent.

Reads chunked/*.jsonl, finds children whose text_hash appears in more
than one case, keeps the copy in the lowest case_id, rewrites affected
files in place, updates _manifest.jsonl, writes _dedup_report.json.

Why post-chunk (not inside chunker):
    The chunker is a pure, per-opinion function (see chunker.py's
    "No I/O, no database" contract). Cross-opinion dedup requires
    state about every other opinion, which would break that contract
    and make chunk_opinion non-deterministic in parallel mode. Running
    dedup as a separate stage preserves chunker purity and lets dedup
    be idempotent, measurable, and skippable.

Why pre-embed:
    Dedup must run BEFORE embed so that dropped children never reach
    the GPU (or the OpenVINO CPU path). Deduping after embed would
    waste embed compute and require rewriting embedded/*.jsonl.

Why exact-hash and not MinHash:
    inspect_data.py measured 8.32% exact-text-hash duplicates on the
    min corpus. Exact-hash mode catches 100% of those with stdlib only
    (no datasketch dependency). MinHash (Jaccard LSH) catches
    near-duplicates — boilerplate with slight variation — but adds a
    dependency and is only justified if this report shows >1% residual
    near-duplicates. Deferred to Phase 2.5 pending a measurement.

Why keep_lowest_case_id:
    Deterministic across runs, independent of processing order.
    'keep_first_seen' would be non-deterministic under parallel
    chunking (which case got processed first is worker-dependent) and
    would make gate_8 flake. Only keep_lowest_case_id is supported.

What this file does NOT do:
    - Touch parents (never indexed; preserved verbatim in each file)
    - Repair intra-case duplicates (gate_6 enforces 0; any found here
      is logged as a warning but not fixed — that's a chunker bug)
    - Guess near-duplicates (exact hash only in v1)
    - Re-normalize text (chunker.text_hash is already canonical)

Interaction with pipeline.load():
    Dedup drops children from case files. If the database was already
    loaded from a pre-dedup run, its chunks table retains the dropped
    rows. The caller MUST either (a) start from an empty database, or
    (b) ensure pipeline.load() deletes chunks not present in the new
    staging set. This file produces correct artifacts; the DB-side
    cleanup is pipeline.load()'s responsibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    utc_now,
)

DEDUP_VERSION = "1"


# ── Errors and report ──────────────────────────────────────────────────


class DedupError(Exception):
    """Raised when dedup cannot proceed (malformed input, manifest drift)."""


@dataclass(frozen=True, slots=True)
class DedupReport:
    """Returned by dedup_chunks(). Also serialized to _dedup_report.json."""

    cases_scanned: int
    cases_rewritten: int
    chunks_scanned: int
    chunks_dropped: int
    duplicate_hashes: int
    intra_case_duplicates: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    dry_run: bool = False


# ── Scanning helpers ───────────────────────────────────────────────────


def _iter_case_files(chunked_dir: Path) -> list[Path]:
    """Return sorted {case_id}.jsonl files; skip _manifest/_failures/_HALTED."""
    return sorted(
        p for p in chunked_dir.glob("*.jsonl") if not p.name.startswith("_") and p.stem.isdigit()
    )


def _extract_hash(row: dict) -> str:
    """Return text_hash from a chunk row; raise if missing or malformed.

    chunker.py v1.4+ always writes text_hash as "sha256:<hex>". Anything
    else means the file was hand-edited or a chunker regression occurred.
    Fail loudly — a silent skip would hide duplicate chunks in retrieval.
    """
    h = row.get("text_hash")
    if not isinstance(h, str) or not h.startswith("sha256:"):
        raise DedupError(
            f"chunk {row.get('chunk_id')!r} missing valid text_hash (got {type(h).__name__}: {h!r})"
        )
    return h


def _scan(
    chunked_dir: Path,
) -> tuple[
    dict[int, list[dict]],  # case_id -> all rows (parents + children)
    dict[str, set[int]],  # text_hash -> set of case_ids
    list[dict],  # intra-case duplicate warnings
]:
    """Read every case file once. Build hash→cases index.

    Raises:
        DedupError: on malformed rows (missing text_hash)
        ValueError: on malformed JSONL (from io_utils.read_jsonl)
    """
    case_rows: dict[int, list[dict]] = {}
    hash_to_cases: dict[str, set[int]] = {}
    intra_case_warnings: list[dict] = []

    for path in _iter_case_files(chunked_dir):
        case_id = int(path.stem)
        rows = list(read_jsonl(path))
        case_rows[case_id] = rows

        seen_in_case: dict[str, int] = {}
        for row in rows:
            if row.get("is_parent"):
                continue
            h = _extract_hash(row)
            seen_in_case[h] = seen_in_case.get(h, 0) + 1
            hash_to_cases.setdefault(h, set()).add(case_id)

        # Intra-case duplicates — chunker regression, warning only
        intra = [h for h, n in seen_in_case.items() if n > 1]
        if intra:
            intra_case_warnings.append({"case_id": case_id, "chunk_hashes": intra})

    return case_rows, hash_to_cases, intra_case_warnings


# ── Winner selection and file rewrite ──────────────────────────────────


def _select_winners(hash_to_cases: dict[str, set[int]]) -> dict[str, int]:
    """Map each cross-case duplicate hash to its winning case_id.

    Policy: keep_lowest_case_id. Deterministic, order-independent.
    """
    winners: dict[str, int] = {}
    for h, cases in hash_to_cases.items():
        if len(cases) > 1:
            winners[h] = min(cases)
    return winners


def _rewrite_case(
    case_id: int,
    rows: list[dict],
    winners: dict[str, int],
) -> tuple[list[dict], int]:
    """Filter a case's rows, dropping children whose hash won elsewhere.

    Returns (kept_rows, dropped_count).
    Parents are always kept — they are never indexed and therefore
    are not dedup candidates.
    """
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        if row.get("is_parent"):
            kept.append(row)
            continue
        h = _extract_hash(row)
        winner = winners.get(h)
        if winner is not None and winner != case_id:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


# ── Public entry point ─────────────────────────────────────────────────


def dedup_chunks(
    chunked_dir: Path,
    *,
    dry_run: bool = False,
) -> DedupReport:
    """Deduplicate cross-case children by exact text_hash.

    Args:
        chunked_dir:  directory containing {case_id}.jsonl files and
                      _manifest.jsonl. Must already exist; missing
                      directory raises DedupError.
        dry_run:      if True, compute the report but do not rewrite
                      any files (useful for `cli dedup --dry-run`).

    Returns:
        DedupReport with counts, elapsed time, and any intra-case
        duplicate warnings.

    Raises:
        DedupError: if chunked_dir or _manifest.jsonl is missing, or
                    if any case file has malformed text_hash.
        ValueError: if any case file has malformed JSONL.

    Side effects (when not dry_run):
        - Rewrites {case_id}.jsonl for every case that lost children
        - Updates _manifest.jsonl entries (chunk_count, embedding_count,
          file_sha256, deduped_at) for rewritten cases
        - Writes _dedup_report.json
        - Does NOT delete cases whose children were all dropped; such
          cases retain only parents (retrieval-inert but harmless)
    """
    started = time.monotonic()

    chunked_dir = Path(chunked_dir)
    if not chunked_dir.is_dir():
        raise DedupError(f"chunked_dir does not exist: {chunked_dir}")

    manifest_path = chunked_dir / "_manifest.jsonl"
    if not manifest_path.is_file():
        raise DedupError(f"missing manifest: {manifest_path}")

    # ── 1. Scan ─────────────────────────────────────────────────────
    case_rows, hash_to_cases, intra_case_warnings = _scan(chunked_dir)

    # Manifest/file agreement — fail fast if they drifted
    manifest_rows = list(read_jsonl(manifest_path))
    manifest_ids = {int(r["case_id"]) for r in manifest_rows}
    file_ids = set(case_rows.keys())
    if manifest_ids != file_ids:
        only_files = sorted(file_ids - manifest_ids)[:5]
        only_manifest = sorted(manifest_ids - file_ids)[:5]
        raise DedupError(
            f"manifest/file mismatch: "
            f"files without manifest={only_files}, "
            f"manifest without files={only_manifest}"
        )

    # ── 2. Select winners for cross-case duplicates ────────────────
    winners = _select_winners(hash_to_cases)

    total_chunks = sum(len(rows) for rows in case_rows.values())
    duplicate_hashes = len(winners)

    # ── 3. No duplicates → write report, return early (idempotency) ─
    if not winners:
        report = DedupReport(
            cases_scanned=len(case_rows),
            cases_rewritten=0,
            chunks_scanned=total_chunks,
            chunks_dropped=0,
            duplicate_hashes=0,
            intra_case_duplicates=intra_case_warnings,
            elapsed_s=round(time.monotonic() - started, 3),
            dry_run=dry_run,
        )
        if not dry_run:
            _write_report(chunked_dir, report)
        return report

    # ── 4. Rewrite affected case files, update manifest in memory ───
    manifest_by_case = {int(r["case_id"]): dict(r) for r in manifest_rows}
    cases_rewritten = 0
    chunks_dropped = 0

    for case_id, rows in case_rows.items():
        kept, dropped = _rewrite_case(case_id, rows, winners)
        if dropped == 0:
            continue

        chunks_dropped += dropped
        cases_rewritten += 1

        if dry_run:
            continue

        case_path = chunked_dir / f"{case_id}.jsonl"
        atomic_write_jsonl(case_path, kept)

        entry = manifest_by_case[case_id]
        entry["chunk_count"] = len(kept)
        entry["embedding_count"] = sum(1 for r in kept if not r.get("is_parent"))
        entry["file_sha256"] = sha256_file(case_path)
        entry["deduped_at"] = utc_now()

    # ── 5. Persist manifest and report (atomic, once each) ──────────
    if not dry_run:
        atomic_write_jsonl(
            manifest_path,
            (manifest_by_case[cid] for cid in sorted(manifest_by_case)),
        )

    report = DedupReport(
        cases_scanned=len(case_rows),
        cases_rewritten=cases_rewritten,
        chunks_scanned=total_chunks,
        chunks_dropped=chunks_dropped,
        duplicate_hashes=duplicate_hashes,
        intra_case_duplicates=intra_case_warnings,
        elapsed_s=round(time.monotonic() - started, 3),
        dry_run=dry_run,
    )
    if not dry_run:
        _write_report(chunked_dir, report)
    return report


# ── Report writer ──────────────────────────────────────────────────────


def _write_report(chunked_dir: Path, report: DedupReport) -> None:
    """Write _dedup_report.json. Structured to mirror validation.json."""
    payload = {
        "dedup_version": DEDUP_VERSION,
        "mode": "exact",
        "policy": "keep_lowest_case_id",
        "totals": {
            "cases_scanned": report.cases_scanned,
            "cases_rewritten": report.cases_rewritten,
            "chunks_scanned": report.chunks_scanned,
            "chunks_dropped": report.chunks_dropped,
            "chunks_kept": report.chunks_scanned - report.chunks_dropped,
        },
        "duplicate_hashes": report.duplicate_hashes,
        "intra_case_duplicates": report.intra_case_duplicates,
        "duplicate_rate": round(report.chunks_dropped / report.chunks_scanned, 4)
        if report.chunks_scanned
        else 0.0,
        "elapsed_s": report.elapsed_s,
        "dry_run": report.dry_run,
        "ran_at": utc_now(),
    }
    atomic_write_json(chunked_dir / "_dedup_report.json", payload)


__all__ = [
    "DEDUP_VERSION",
    "DedupError",
    "DedupReport",
    "dedup_chunks",
]
