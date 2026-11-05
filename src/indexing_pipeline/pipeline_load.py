"""Load-side stages — embed, load, validate, index, drop_indexes.

LoadMixin supplies these stages to IndexingPipeline (composed in
pipeline.py). At runtime LoadMixin inherits from object so
`IndexingPipeline(LoadMixin, IndexingPipelineBase)` produces a clean MRO.
At type-check time LoadMixin inherits from IndexingPipelineBase (via the
TYPE_CHECKING branch below), so basedpyright resolves every shared
attribute and method without requiring `self: IndexingPipelineBase`
annotations (which basedpyright rejects for sibling classes).

v2.7 changes:
  - load() uses loaded_ids (surviving cases from chunk_manifest) instead
    of plan_ids (all requested cases). Cases dropped by quality_filter in
    chunk() are NOT in chunk_manifest, so loading by plan_ids would
    insert cases with no chunks — inflating the cases table and
    distorting graph endpoint filtering.
  - gate_9 threshold: 0.9999 → 0.99 (measured cross-backend T4-vs-CPU
    cosine 0.9966; 0.9999 was a same-backend threshold, wrong for
    cross-backend comparison).
  - load() logs drop count at start ("loading X/Y after quality filter
    dropped Z") so the number is visible without parsing the drops file.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import IndexingPipelineBase
else:
    # Runtime placeholder: LoadMixin does not need a real parent.
    # IndexingPipeline provides the MRO composition.
    IndexingPipelineBase = object  # type: ignore[misc,assignment]

import numpy as np

from . import chunker, db
from .embedder import EMBEDDING_DIMENSIONS
from .filters import is_noise_chunk, normalize_precedential_status
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    json_safe,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .pipeline_helpers import (
    EMBED_CASES_PER_BATCH,
    PIPELINE_STATE_KEY,
    PipelineError,
    StageReport,
)

logger = logging.getLogger("indexing_pipeline")


class LoadMixin(IndexingPipelineBase):  # type: ignore[misc]
    """Stages from embedding onward: embed, load, validate, index, drop-index."""

    # ── Stage: embed — batched, per-case fallback, device-aware ─────

    def embed(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()

            if self._is_halted("embed") and not self.config.force:
                raise PipelineError("embed", "halted")
            if self.config.force:
                self._clear_halted("embed")

            chunk_manifest = self._load_manifest(self.config.chunked_dir)
            if not chunk_manifest:
                raise PipelineError("embed", "chunked manifest is empty")

            if self.config.dry_run:
                logger.info(
                    "embed dry-run: %d cases, device=%s",
                    len(chunk_manifest),
                    self.config.device,
                )
                return StageReport("embed", 0, len(chunk_manifest), 0, time.monotonic() - started)

            self.config.embedded_dir.mkdir(parents=True, exist_ok=True)
            previous = self._load_manifest(self.config.embedded_dir)
            failures = self._get_failures(self.config.embedded_dir)

            new_manifest: dict[int, dict] = {}
            processed = skipped = failed = 0
            case_ids = sorted(chunk_manifest)

            logger.info(
                "embed starting: %d cases, device=%s",
                len(case_ids),
                self.config.device,
            )
            embedder = self._get_embedder()
            logger.info("embed model loaded, beginning batched inference")

            for batch_start in range(0, len(case_ids), EMBED_CASES_PER_BATCH):
                case_batch = case_ids[batch_start : batch_start + EMBED_CASES_PER_BATCH]

                pending: list[tuple[int, dict, list[dict]]] = []
                for case_id in case_batch:
                    chunk_entry = chunk_manifest[case_id]
                    if not self.config.force and self._is_valid_embed_output(
                        case_id, chunk_entry, previous.get(case_id)
                    ):
                        new_manifest[case_id] = previous[case_id]
                        skipped += 1
                        continue

                    try:
                        chunk_path = self._chunked_file(case_id)
                        if not chunk_path.is_file():
                            raise FileNotFoundError(chunk_path)
                        if sha256_file(chunk_path) != chunk_entry["file_sha256"]:
                            raise PipelineError("embed", f"chunk hash mismatch {case_id}")
                        rows = list(read_jsonl(chunk_path))
                        children = [r for r in rows if not bool(r["is_parent"])]
                        if len(children) != int(chunk_entry["embedding_count"]):
                            raise PipelineError(
                                "embed",
                                f"child count mismatch {case_id}: "
                                f"{len(children)} != {chunk_entry['embedding_count']}",
                            )
                        pending.append((case_id, chunk_entry, children))
                    except Exception as exc:
                        failed += 1
                        processed += 1
                        failures[case_id] = {
                            "case_id": case_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "stage": "embed",
                            "ts": utc_now(),
                        }

                if not pending:
                    if processed and processed % 10 == 0:
                        logger.info(
                            "embed progress: %d/%d, skipped=%d, failed=%d",
                            processed,
                            len(case_ids),
                            skipped,
                            failed,
                        )
                    continue

                all_texts: list[str] = []
                slices: list[tuple[int, int]] = []
                for _, _, children in pending:
                    slice_start = len(all_texts)
                    all_texts.extend(str(r["text"]) for r in children)
                    slices.append((slice_start, len(all_texts)))

                batch_ok = False
                all_vecs = None
                try:
                    all_vecs = embedder.embed_documents(all_texts)
                    if all_vecs.shape != (len(all_texts), EMBEDDING_DIMENSIONS):
                        raise RuntimeError(
                            f"batch returned {all_vecs.shape}, "
                            f"expected ({len(all_texts)}, {EMBEDDING_DIMENSIONS})"
                        )
                    batch_ok = True
                except Exception as exc:
                    logger.warning(
                        "batch embed failed (%s); retrying per-case to isolate offenders",
                        type(exc).__name__,
                    )

                if batch_ok and all_vecs is not None:
                    for (case_id, chunk_entry, children), (
                        slice_start,
                        slice_end,
                    ) in zip(pending, slices, strict=True):
                        try:
                            vectors = all_vecs[slice_start:slice_end]
                            if vectors.shape != (
                                len(children),
                                EMBEDDING_DIMENSIONS,
                            ):
                                raise RuntimeError(f"shape mismatch {case_id}: {vectors.shape}")
                            output_rows = [
                                {"chunk_id": r["chunk_id"], "vector": v.tolist()}
                                for r, v in zip(children, vectors, strict=True)
                            ]
                            out_path = self._embedded_file(case_id)
                            atomic_write_jsonl(out_path, output_rows)
                            new_manifest[case_id] = {
                                "case_id": case_id,
                                "chunk_count": len(output_rows),
                                "model_version": self.config.embedder_version,
                                "device": self.config.device,
                                "chunked_file_sha256": chunk_entry["file_sha256"],
                                "file_sha256": sha256_file(out_path),
                                "completed_at": utc_now(),
                            }
                            failures.pop(case_id, None)
                        except Exception as exc:
                            failed += 1
                            failures[case_id] = {
                                "case_id": case_id,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "stage": "embed",
                                "ts": utc_now(),
                            }
                        processed += 1
                else:
                    for case_id, chunk_entry, children in pending:
                        try:
                            per_vecs = embedder.embed_documents([str(r["text"]) for r in children])
                            if per_vecs.shape != (
                                len(children),
                                EMBEDDING_DIMENSIONS,
                            ):
                                raise RuntimeError(f"shape mismatch {case_id}: {per_vecs.shape}")
                            output_rows = [
                                {"chunk_id": r["chunk_id"], "vector": v.tolist()}
                                for r, v in zip(children, per_vecs, strict=True)
                            ]
                            out_path = self._embedded_file(case_id)
                            atomic_write_jsonl(out_path, output_rows)
                            new_manifest[case_id] = {
                                "case_id": case_id,
                                "chunk_count": len(output_rows),
                                "model_version": self.config.embedder_version,
                                "device": self.config.device,
                                "chunked_file_sha256": chunk_entry["file_sha256"],
                                "file_sha256": sha256_file(out_path),
                                "completed_at": utc_now(),
                            }
                            failures.pop(case_id, None)
                        except Exception as case_exc:
                            failed += 1
                            failures[case_id] = {
                                "case_id": case_id,
                                "error": f"{type(case_exc).__name__}: {case_exc}",
                                "traceback": traceback.format_exc(),
                                "stage": "embed",
                                "ts": utc_now(),
                            }
                        processed += 1

                logger.info(
                    "embed progress: %d/%d, skipped=%d, failed=%d",
                    processed,
                    len(case_ids),
                    skipped,
                    failed,
                )
                self._persist_embed_state(new_manifest, failures)

                if self._check_failure_threshold(failed, processed):
                    reason = f"failure threshold exceeded: {failed}/{processed}"
                    self._write_halted("embed", "systemic", failed, processed)
                    raise PipelineError(
                        "embed",
                        "halted",
                        StageReport(
                            "embed",
                            processed,
                            skipped,
                            failed,
                            time.monotonic() - started,
                            reason,
                        ),
                    )

            self._persist_embed_state(new_manifest, failures)

            if self._check_failure_threshold(failed, processed):
                reason = f"failure threshold exceeded: {failed}/{processed}"
                self._write_halted("embed", "systemic", failed, processed)
                raise PipelineError(
                    "embed",
                    "halted",
                    StageReport(
                        "embed",
                        processed,
                        skipped,
                        failed,
                        time.monotonic() - started,
                        reason,
                    ),
                )

            orphans = self._cleanup_orphans(self.config.embedded_dir, set(new_manifest.keys()))
            if orphans:
                logger.info("embed cleaned %d orphan files", orphans)

            logger.info(
                "embed complete: %d processed, %d skipped, %d failed, %d orphans (device=%s)",
                processed,
                skipped,
                failed,
                orphans,
                self.config.device,
            )
            return StageReport("embed", processed, skipped, failed, time.monotonic() - started)

    # ── Load helpers ────────────────────────────────────────────────

    def _load_clusters(self) -> dict[int, dict]:
        result: dict[int, dict] = {}

        for row in self._iter_parquet(self.config.clusters_file):
            cid = row.get("id")
            if cid is None:
                continue
            cid = int(cid)
            if cid in result:
                raise PipelineError("load", f"duplicate cluster id: {cid}")
            result[cid] = row

        return result

    @staticmethod
    def _join_cases(corpus_row: dict, cluster_row: dict | None) -> dict:
        cluster_row = cluster_row or {}

        def choose(key: str) -> Any:
            v = corpus_row.get(key)
            return v if v is not None else cluster_row.get(key)

        reserved = {
            "case_id",
            "cl_opinion_id",
            "cluster_id",
            "text",
            "text_kind",
            "metadata",
        }
        metadata = json_safe({k: v for k, v in cluster_row.items() if k not in reserved})
        cid = corpus_row.get("cluster_id")

        return {
            "case_id": int(corpus_row["cl_opinion_id"]),
            "cluster_id": int(cid) if cid is not None else None,
            "court": choose("court") or "",
            "date": choose("date"),
            "title": choose("title") or "",
            "citation": choose("citation") or "",
            "opinion_type": str(choose("opinion_type") or ""),
            "jurisdiction": choose("jurisdiction"),
            "doc_type": choose("doc_type"),
            "case_name": choose("case_name"),
            "case_name_short": choose("case_name_short"),
            "precedential_status": normalize_precedential_status(choose("precedential_status")),
            "nature_of_suit": choose("nature_of_suit"),
            "citation_count": choose("citation_count"),
            "judges": choose("judges"),
            "metadata": metadata,
        }

    def _merge_chunks_and_vectors(self, case_id: int) -> Iterator[dict]:
        chunk_rows = list(read_jsonl(self._chunked_file(case_id)))
        vector_rows = list(read_jsonl(self._embedded_file(case_id)))

        vec_map: dict[str, list[float]] = {}
        for row in vector_rows:
            cid = str(row["chunk_id"])
            vector = row.get("vector")
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
                raise PipelineError("load", f"invalid vector for chunk {cid}")
            vec_map[cid] = [float(v) for v in vector]

        for row in chunk_rows:
            is_parent = bool(row["is_parent"])
            chunk_id = str(row["chunk_id"])
            out = dict(row)
            out["embedding"] = None
            out["embedder_version"] = None
            if not is_parent:
                vector = vec_map.pop(chunk_id, None)
                if vector is None:
                    raise PipelineError("load", f"missing embedding for {chunk_id}")
                out["embedding"] = vector
                out["embedder_version"] = self.config.embedder_version
            yield out

        if vec_map:
            raise PipelineError(
                "load",
                f"orphan embeddings for case {case_id}: {len(vec_map)} "
                f"(re-run `embed` after `dedup`)",
            )

    # ── Stage: load ─────────────────────────────────────────────────

    def load(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            self._check_inputs(include_metadata=True)

            chunk_manifest = self._load_manifest(self.config.chunked_dir)
            embed_manifest = self._load_manifest(self.config.embedded_dir)

            if not chunk_manifest:
                raise PipelineError("load", "chunked manifest is empty (all cases dropped?)")
            if set(chunk_manifest) != set(embed_manifest):
                raise PipelineError("load", "chunked and embedded case IDs differ")

            if any(read_jsonl(self.config.chunked_dir / "_failures.jsonl")):
                raise PipelineError("load", "precondition: unresolved chunking failures")
            if any(read_jsonl(self.config.embedded_dir / "_failures.jsonl")):
                raise PipelineError("load", "precondition: unresolved embedding failures")

            for case_id, chunk_entry in chunk_manifest.items():
                chunk_path = self._chunked_file(case_id)
                if not chunk_path.is_file():
                    raise PipelineError("load", f"missing chunk file for {case_id}")
                if chunk_entry.get("file_sha256") != sha256_file(chunk_path):
                    raise PipelineError("load", f"chunk file hash mismatch for {case_id}")
                if not self._is_valid_embed_output(case_id, chunk_entry, embed_manifest[case_id]):
                    raise PipelineError(
                        "load",
                        f"invalid embed output for {case_id} "
                        f"(re-run `embed` after chunk/dedup changes)",
                    )

            if self.config.dry_run:
                logger.info("load dry-run: %d cases", len(chunk_manifest))
                return StageReport("load", 0, len(chunk_manifest), 0, time.monotonic() - started)

            database = self._get_database()
            database.ensure_schema()

            fingerprint = self._dataset_fingerprint()
            state = database.get_pipeline_state(PIPELINE_STATE_KEY)

            if (
                state
                and state.get("dataset_fingerprint") == fingerprint
                and state.get("chunker_version") == self.config.chunker_version
                and state.get("embedder_version") == self.config.embedder_version
            ):
                if database.index_exists() and database.index_is_valid():
                    logger.info("load skipped: unchanged dataset, index valid")
                    return StageReport(
                        "load", 0, len(chunk_manifest), 0, time.monotonic() - started
                    )
                if not database.index_exists():
                    logger.info("load skipped: unchanged dataset, index absent")
                    return StageReport(
                        "load", 0, len(chunk_manifest), 0, time.monotonic() - started
                    )

            if database.index_exists():
                raise PipelineError(
                    "load",
                    "precondition: index exists; run 'drop-index' then 'load' or 'all --force'",
                )

            # v2.7: use loaded_ids (surviving cases) not plan_ids (all
            # requested cases). Cases dropped by quality_filter in chunk()
            # are NOT in chunk_manifest, so loading by plan_ids would load
            # cases with no chunks — inflating the cases table and breaking
            # graph endpoint filtering (edges would need endpoints that
            # survive the filter, which they don't if one endpoint is
            # dropped).
            plan = self._load_plan()  # validates plan.jsonl exists, non-empty
            if not plan:
                raise PipelineError("load", "precondition: plan is empty or missing")

            loaded_ids = set(chunk_manifest.keys())
            dropped_count = len(plan) - len(loaded_ids)
            if dropped_count > 0:
                logger.info(
                    "load: %d case(s) dropped by quality filter during chunk (loading %d/%d)",
                    dropped_count,
                    len(loaded_ids),
                    len(plan),
                )

            deleted = database.delete_stale_cases(list(loaded_ids))
            if deleted:
                logger.info(
                    "deleted %d stale cases (cascades to edges and chunks)",
                    deleted,
                )

            clusters = self._load_clusters()

            def case_rows() -> Iterator[dict]:
                for record in self._iter_opinions(loaded_ids):
                    cid = record.get("cluster_id")
                    cluster = clusters.get(int(cid)) if cid is not None else None
                    yield self._join_cases(record, cluster)

            case_count = database.load_cases(case_rows())

            loaded_ids_set = loaded_ids

            def edge_rows() -> Iterator[dict]:
                for row in self._iter_parquet(self.config.edges_file):
                    source = int(row["citing_opinion_id"])
                    target = int(row["cited_opinion_id"])
                    if source not in loaded_ids_set or target not in loaded_ids_set:
                        continue
                    yield {
                        "source_id": source,
                        "target_id": target,
                        "relation": "CITES",
                        "depth": int(row["depth"]) if row.get("depth") is not None else None,
                    }

            database.load_edges(edge_rows())

            def chunk_rows() -> Iterator[dict]:
                for case_id in sorted(chunk_manifest):
                    yield from self._merge_chunks_and_vectors(case_id)

            staged = database.stage_chunks(chunk_rows())
            upserted, deleted_chunks = database.merge_chunks()

            counts = database.row_counts()
            database.set_pipeline_state(
                PIPELINE_STATE_KEY,
                {
                    "dataset_fingerprint": fingerprint,
                    "chunker_version": self.config.chunker_version,
                    "embedder_version": self.config.embedder_version,
                    "filters_version": self.config.filters_version,
                    "dedup_version": self.config.dedup_version,
                    "case_count": counts["cases"],
                    "edge_count": counts["case_edges"],
                    "chunk_count": counts["chunks"],
                    "staged_chunk_count": staged,
                    "upserted_chunk_count": upserted,
                    "deleted_stale_chunk_count": deleted_chunks,
                    "dropped_by_quality_filter": dropped_count,
                    "loaded_at": utc_now(),
                },
            )

            logger.info(
                "load complete: cases=%d (affected=%d, dropped=%d), edges=%d, "
                "staged=%d, upserted=%d, deleted_stale=%d, chunks=%d, "
                "children=%d, parents=%d, orphan_parents=%d",
                counts["cases"],
                case_count,
                dropped_count,
                counts["case_edges"],
                staged,
                upserted,
                deleted_chunks,
                counts["chunks"],
                counts["children"],
                counts["parents"],
                counts["orphan_parents"],
            )

            return StageReport("load", staged, 0, 0, time.monotonic() - started)

    # ── Stage: validate — determinism + DB gates ───────────────────

    def _determinism_check(self, sample_size: int = 20) -> db.GateResult:
        """Re-chunk N cases and verify stored chunks are a subset of re-chunk.

        Subset (not equality) because dedup.py may have removed cross-case
        duplicates from chunked/*.jsonl. Re-chunking reproduces the full
        pre-dedup set; stored must be a subset of that set.
        """
        chunk_manifest = self._load_manifest(self.config.chunked_dir)
        candidates = sorted(chunk_manifest)[:sample_size]
        if not candidates:
            return db.GateResult(
                name="gate_8_determinism",
                passed=False,
                value="no chunked opinions available",
                threshold="stored ⊆ re-chunked",
            )

        tokenize = self._get_tokenizer()
        noise = is_noise_chunk if self.config.noise_filter_enabled else None
        sample = set(candidates)
        checked = 0
        mismatches: list[dict] = []

        for record in self._iter_opinions(sample):
            case_id = int(record["cl_opinion_id"])
            if case_id not in sample:
                continue
            recomputed = chunker.chunk_opinion(record, tokenize=tokenize, is_noise=noise)
            expected = list(read_jsonl(self._chunked_file(case_id)))

            recomputed_sig = {
                (c.chunk_id, c.text_hash, c.parent_id, c.is_parent) for c in recomputed
            }
            stored_sig = {
                (
                    r["chunk_id"],
                    r["text_hash"],
                    r.get("parent_id"),
                    bool(r["is_parent"]),
                )
                for r in expected
            }
            missing = stored_sig - recomputed_sig
            if missing:
                mismatches.append(
                    {
                        "case_id": case_id,
                        "missing": [list(m)[:2] for m in list(missing)[:5]],
                    }
                )
            else:
                checked += 1

        return db.GateResult(
            name="gate_8_determinism",
            passed=not mismatches,
            value={
                "checked": checked,
                "expected": len(candidates),
                "mismatches": mismatches[:20],
            },
            threshold=f"stored ⊆ re-chunked on {len(candidates)} sampled cases",
        )

    def _embedding_determinism_check(self, sample_size: int = 3) -> db.GateResult:
        """Re-embed N cases and verify cosine >= threshold.

        Cross-backend comparison: cases were originally embedded on Kaggle
        T4 (FP16 tensor cores), re-embedded here on OpenVINO CPU (FP16
        weights converted to FP32 at matmul). Different numerical paths,
        same model. Measured cosine ~0.9966 on real data. Threshold 0.99
        catches genuine regressions (corrupted model would drop below
        0.99) while tolerating expected cross-backend FP16 drift.
        """
        chunk_manifest = self._load_manifest(self.config.chunked_dir)
        candidates = sorted(chunk_manifest)[:sample_size]
        if not candidates:
            return db.GateResult(
                name="gate_9_embedding_determinism",
                passed=False,
                value="no chunked opinions",
                threshold="cosine >= 0.99 on re-embed",
            )

        embedder = self._get_embedder()
        worst = 1.0
        checked = 0
        for case_id in candidates:
            embed_path = self._embedded_file(case_id)
            if not embed_path.is_file():
                continue
            chunk_rows = list(read_jsonl(self._chunked_file(case_id)))
            children = [r for r in chunk_rows if not r["is_parent"]]
            stored = list(read_jsonl(embed_path))
            if len(stored) != len(children):
                return db.GateResult(
                    name="gate_9_embedding_determinism",
                    passed=False,
                    value=(f"case {case_id}: {len(stored)} vectors != {len(children)} children"),
                    threshold="cosine >= 0.99",
                )
            if not children:
                checked += 1
                continue
            new_vecs = embedder.embed_documents([r["text"] for r in children])
            old_vecs = np.asarray([r["vector"] for r in stored], dtype=np.float32)
            cosines = (new_vecs * old_vecs).sum(axis=1)
            worst = min(worst, float(cosines.min()))
            checked += 1

        return db.GateResult(
            name="gate_9_embedding_determinism",
            passed=worst >= 0.99,
            value={"min_cosine": worst, "checked_cases": checked},
            threshold="cosine >= 0.99 (cross-backend T4/OpenVINO FP16)",
        )

    def validate(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            if self.config.dry_run:
                logger.info("validate dry-run")
                return StageReport("validate", 0, 0, 0, time.monotonic() - started)

            database = self._get_database()
            database.ensure_schema()
            determinism = self._determinism_check()
            embedding_determinism = self._embedding_determinism_check()
            report = database.validation_report(
                determinism_gate=determinism,
                extra_gates={"gate_9_embedding_determinism": embedding_determinism},
            )

            self.config.reports_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "ok": report.ok,
                "measured_at": report.measured_at.isoformat(),
                "gates": {
                    n: {
                        "passed": g.passed,
                        "value": g.value,
                        "threshold": g.threshold,
                        "note": g.note,
                    }
                    for n, g in report.gates.items()
                },
                "row_counts": report.row_counts,
                "dataset_fingerprint": self._dataset_fingerprint(),
            }
            atomic_write_json(self.config.reports_dir / "validation.json", payload)

            if not report.ok:
                failed = [n for n, g in report.gates.items() if not g.passed]
                raise PipelineError(
                    "validate",
                    "gate_failed: " + ", ".join(failed),
                    StageReport(
                        "validate",
                        1,
                        0,
                        len(failed),
                        time.monotonic() - started,
                    ),
                )
            return StageReport("validate", 1, 0, 0, time.monotonic() - started)

    # ── Stage: index ────────────────────────────────────────────────

    def index(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            if self.config.dry_run:
                logger.info("index dry-run")
                return StageReport("index", 0, 0, 0, time.monotonic() - started)

            database = self._get_database()
            database.ensure_schema()

            validation_path = self.config.reports_dir / "validation.json"
            if not validation_path.is_file():
                raise PipelineError("index", "precondition: validation.json missing")

            validation = json.loads(validation_path.read_text())
            current = self._dataset_fingerprint()
            if not validation.get("ok") or validation.get("dataset_fingerprint") != current:
                raise PipelineError(
                    "index",
                    "precondition: validation failed or fingerprint mismatch",
                )

            state = database.get_pipeline_state(PIPELINE_STATE_KEY)
            if not state or state.get("dataset_fingerprint") != current:
                raise PipelineError(
                    "index",
                    "precondition: dataset not loaded or fingerprint mismatch",
                )

            if self.config.force:
                database.drop_indexes()

            index_report: dict[str, Any] = dict(database.build_indexes())
            index_report["built_at"] = utc_now()
            index_report["dataset_fingerprint"] = current
            index_report["row_counts"] = database.row_counts()

            self.config.reports_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.config.reports_dir / "index_report.json", index_report)

            return StageReport("index", len(index_report), 0, 0, time.monotonic() - started)

    # ── Stage: drop-index ───────────────────────────────────────────

    def drop_indexes(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            if self.config.dry_run:
                logger.info("drop-index dry-run")
                return StageReport("drop-index", 0, 0, 0, time.monotonic() - started)

            database = self._get_database()
            database.ensure_schema()

            if not database.index_exists():
                logger.info("no indexes present; nothing to drop")
                return StageReport("drop-index", 0, 0, 0, time.monotonic() - started)

            database.drop_indexes()
            logger.info("indexes dropped")
            return StageReport("drop-index", 1, 0, 0, time.monotonic() - started)


__all__ = ["LoadMixin"]
