"""Orchestration — IndexingPipeline stages. Imports helpers, chunker, db, embedder.

All constants, Config, errors, and worker functions live in pipeline_helpers.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import json
import logging
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pyarrow.parquet as pq

from . import chunker, db
from .embedder import EMBEDDING_DIMENSIONS, Embedder
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    json_safe,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now,
    write_jsonl_from_dict,
)
from .pipeline_helpers import (
    EMBED_CASES_PER_BATCH,
    FAILURE_ABSOLUTE,
    FAILURE_CHECK_EVERY,
    FAILURE_RATE,
    PIPELINE_STATE_KEY,
    Config,
    PipelineError,
    PlanEntry,
    StageReport,
    chunk_worker,
    resolve_corpus_rows,
    worker_init,
)

logger = logging.getLogger("indexing_pipeline")


class IndexingPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config.validated()
        self._embedder: Any = None
        self._database: db.Database | None = None
        self._tokenizer_fn: Callable[[str], int] | None = None
        self._closed = False
        self._lock_path = self.config.data_dir / ".indexing.lock"

    def __enter__(self) -> IndexingPipeline:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        if self._embedder is not None:
            try:
                self._embedder.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._embedder = None
        if self._database is not None:
            try:
                self._database.close()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._database = None
        if errors:
            raise PipelineError("close", "; ".join(str(e) for e in errors))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("pipeline is closed")

    # ── Backend resolution ───────────────────────────────────────────

    def _get_embedder(self) -> Any:
        self._require_open()
        if self._embedder is None:
            if self.config.device.startswith("cuda"):
                from .embedder import TorchEmbedder

                logger.info(
                    "loading TorchEmbedder device=%s model_dir=%s",
                    self.config.device,
                    self.config.gpu_model_dir,
                )
                self._embedder = TorchEmbedder(
                    model_dir=self.config.gpu_model_dir,
                    max_seq_length=self.config.max_seq_length,
                    device=self.config.device,
                    batch_size=max(self.config.embed_batch, 32),
                )
            else:
                logger.info(
                    "loading OpenVINO Embedder device=cpu model_dir=%s",
                    self.config.model_dir,
                )
                self._embedder = Embedder(
                    model_dir=self.config.model_dir,
                    max_seq_length=self.config.max_seq_length,
                    threads=self.config.inference_threads,
                    batch_size=self.config.embed_batch,
                    device="CPU",
                )
        return self._embedder

    def _get_database(self) -> db.Database:
        self._require_open()
        if not self.config.db_dsn:
            raise PipelineError(
                "database",
                "DATABASE_URL is required for this stage (load, validate, index, "
                "drop-index). The embed stage runs without a database.",
            )
        if self._database is None:
            self._database = db.Database(
                dsn=self.config.db_dsn,
                pool_min=self.config.pool_min,
                pool_max=self.config.pool_max,
            )
        return self._database

    def _get_tokenizer(self) -> Callable[[str], int]:
        if self._tokenizer_fn is None:
            from transformers import AutoTokenizer

            cpu_tok = self.config.model_dir / "tokenizer.json"
            gpu_tok = self.config.gpu_model_dir / "tokenizer.json"
            if cpu_tok.is_file():
                tok_dir = self.config.model_dir
            elif gpu_tok.is_file():
                tok_dir = self.config.gpu_model_dir
            else:
                raise PipelineError(
                    "chunk",
                    f"no tokenizer.json under {self.config.model_dir} "
                    f"or {self.config.gpu_model_dir}",
                )
            tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
            self._tokenizer_fn = lambda t: len(
                tok.encode(t, add_special_tokens=True, truncation=False)
            )
            logger.info("chunker tokenizer loaded from %s", tok_dir)
        return self._tokenizer_fn

    # ── Locking ──────────────────────────────────────────────────────

    @contextmanager
    def _stage_lock(self) -> Iterator[None]:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineError("lock", "another pipeline process is active") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ── Input checks ─────────────────────────────────────────────────

    def _check_inputs(self, *, include_metadata: bool = True) -> None:
        required = [self.config.corpus_file]
        if include_metadata:
            required.extend([self.config.clusters_file, self.config.edges_file])
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)

    # ── Parquet iteration ────────────────────────────────────────────

    def _iter_opinions(self, keep_ids: set[int] | None = None) -> Iterator[dict]:
        parquet = pq.ParquetFile(self.config.corpus_file)
        required = {"cl_opinion_id", "text", "text_kind", "opinion_type"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise PipelineError("manifest", f"corpus missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=self.config.batch_size, use_threads=True):
            for row in batch.to_pylist():
                if keep_ids is None or int(row["cl_opinion_id"]) in keep_ids:
                    yield row

    def _iter_parquet(self, path) -> Iterator[dict]:
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=self.config.batch_size, use_threads=True):
            yield from batch.to_pylist()

    def _read_all_case_ids(self) -> list[int]:
        parquet = pq.ParquetFile(self.config.corpus_file)
        ids: list[int] = []
        for batch in parquet.iter_batches(
            batch_size=self.config.batch_size,
            columns=["cl_opinion_id"],
            use_threads=True,
        ):
            for value in batch.column(0).to_pylist():
                ids.append(int(value))
        return ids

    # ── Manifest helpers ─────────────────────────────────────────────

    def _load_plan(self) -> dict[int, PlanEntry]:
        result: dict[int, PlanEntry] = {}
        for row in read_jsonl(self.config.manifest_dir / "plan.jsonl"):
            case_id = int(row["case_id"])
            if case_id in result:
                raise PipelineError("manifest", f"duplicate case_id in plan: {case_id}")
            cid = row.get("cluster_id")
            result[case_id] = PlanEntry(
                case_id=case_id,
                source_text_hash=str(row["source_text_hash"]),
                text_kind=str(row.get("text_kind") or ""),
                cluster_id=int(cid) if cid is not None else None,
            )
        return result

    def _load_manifest(self, directory) -> dict[int, dict]:
        result: dict[int, dict] = {}
        for row in read_jsonl(directory / "_manifest.jsonl"):
            case_id = int(row["case_id"])
            if case_id in result:
                raise PipelineError("manifest", f"duplicate manifest entry: {case_id}")
            result[case_id] = row
        return result

    def _load_manifest_meta(self) -> dict | None:
        rows = list(read_jsonl(self.config.manifest_dir / "_manifest.jsonl"))
        return rows[0] if rows else None

    def _get_failures(self, directory) -> dict[int, dict]:
        return {int(row["case_id"]): row for row in read_jsonl(directory / "_failures.jsonl")}

    def _chunked_file(self, case_id: int):
        return self.config.chunked_dir / f"{case_id}.jsonl"

    def _embedded_file(self, case_id: int):
        return self.config.embedded_dir / f"{case_id}.jsonl"

    # ── Skip logic ───────────────────────────────────────────────────

    def _is_valid_chunk_output(self, case_id: int, entry: dict | None, plan: PlanEntry) -> bool:
        if not entry:
            return False
        path = self._chunked_file(case_id)
        if not path.is_file():
            return False
        return (
            entry.get("source_text_hash") == plan.source_text_hash
            and entry.get("chunker_version") == self.config.chunker_version
            and entry.get("file_sha256") == sha256_file(path)
        )

    def _is_valid_embed_output(
        self, case_id: int, chunk_entry: dict, embed_entry: dict | None
    ) -> bool:
        if not embed_entry:
            return False
        path = self._embedded_file(case_id)
        if not path.is_file():
            return False
        return (
            int(embed_entry.get("chunk_count", -1)) == int(chunk_entry.get("embedding_count", -2))
            and embed_entry.get("chunked_file_sha256") == chunk_entry.get("file_sha256")
            and embed_entry.get("model_version") == self.config.embedder_version
            and embed_entry.get("file_sha256") == sha256_file(path)
        )

    # ── Halt handling ────────────────────────────────────────────────

    def _is_halted(self, stage: str) -> bool:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir
        return (directory / "_HALTED").exists()

    def _clear_halted(self, stage: str) -> None:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir
        try:
            (directory / "_HALTED").unlink()
        except FileNotFoundError:
            pass

    def _write_halted(self, stage: str, reason: str, failed: int, processed: int) -> None:
        directory = self.config.chunked_dir if stage == "chunk" else self.config.embedded_dir
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            directory / "_HALTED",
            {
                "reason": reason,
                "failed": failed,
                "processed": processed,
                "threshold_absolute": FAILURE_ABSOLUTE,
                "threshold_rate": FAILURE_RATE,
                "ts": utc_now(),
            },
        )

    @staticmethod
    def _check_failure_threshold(failed: int, processed: int) -> bool:
        if failed <= 0:
            return False
        if failed > FAILURE_ABSOLUTE:
            return True
        if processed <= 0:
            return False
        return failed / processed > FAILURE_RATE

    def _persist_chunk_state(self, manifest: dict[int, dict], failures: dict[int, dict]) -> None:
        write_jsonl_from_dict(self.config.chunked_dir / "_manifest.jsonl", manifest)
        write_jsonl_from_dict(self.config.chunked_dir / "_failures.jsonl", failures)

    def _persist_embed_state(self, manifest: dict[int, dict], failures: dict[int, dict]) -> None:
        write_jsonl_from_dict(self.config.embedded_dir / "_manifest.jsonl", manifest)
        write_jsonl_from_dict(self.config.embedded_dir / "_failures.jsonl", failures)

    @staticmethod
    def _cleanup_orphans(directory, keep_ids: set[int]) -> int:
        removed = 0
        if not directory.is_dir():
            return 0
        for path in directory.iterdir():
            if path.suffix != ".jsonl" or path.name.startswith("_"):
                continue
            if not path.stem.isdigit():
                continue
            if int(path.stem) not in keep_ids:
                path.unlink()
                removed += 1
        return removed

    # ── Fingerprint ──────────────────────────────────────────────────

    def _dataset_fingerprint(self) -> str:
        plan_path = self.config.manifest_dir / "plan.jsonl"
        plan_rows = sum(1 for _ in read_jsonl(plan_path)) if plan_path.exists() else 0
        components: dict[str, Any] = {
            str(p.relative_to(self.config.data_dir)): sha256_file(p)
            for p in (self.config.corpus_file, self.config.clusters_file, self.config.edges_file)
        }
        components["chunker_version"] = self.config.chunker_version
        components["embedder_version"] = self.config.embedder_version
        components["corpus_rows_effective"] = plan_rows
        return sha256_text(json.dumps(components, sort_keys=True))

    # ════════════════════════════════════════════════════════════════
    # Stage: manifest
    # ════════════════════════════════════════════════════════════════

    def manifest(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            self._check_inputs(include_metadata=False)

            source_hash = sha256_file(self.config.corpus_file)
            requested = resolve_corpus_rows(self.config.corpus_rows)

            all_ids = self._read_all_case_ids()
            effective = min(requested, len(all_ids))
            if effective < requested:
                logger.warning(
                    "corpus_rows requested %d, corpus has only %d",
                    requested,
                    effective,
                )

            meta = self._load_manifest_meta()
            plan_path = self.config.manifest_dir / "plan.jsonl"

            if (
                not self.config.force
                and meta is not None
                and meta.get("source_file_hash") == source_hash
                and int(meta.get("row_count", -1)) == len(all_ids)
                and int(meta.get("corpus_rows_effective", -1)) == effective
                and plan_path.is_file()
            ):
                logger.info("manifest skipped: unchanged (effective=%d)", effective)
                return StageReport("manifest", 0, effective, 0, time.monotonic() - started)

            if self.config.dry_run:
                logger.info(
                    "manifest dry-run: would build %d cases (requested=%d, corpus=%d)",
                    effective,
                    requested,
                    len(all_ids),
                )
                return StageReport("manifest", 0, effective, 0, time.monotonic() - started)

            self.config.manifest_dir.mkdir(parents=True, exist_ok=True)
            keep_ids = set(sorted(all_ids)[:effective])

            plan_rows: list[dict] = []
            for record in self._iter_opinions(keep_ids):
                case_id = int(record["cl_opinion_id"])
                cid = record.get("cluster_id")
                plan_rows.append(
                    {
                        "case_id": case_id,
                        "source_text_hash": sha256_text(record.get("text")),
                        "text_kind": str(record.get("text_kind") or ""),
                        "cluster_id": int(cid) if cid is not None else None,
                    }
                )

            if len(plan_rows) != effective:
                raise PipelineError(
                    "manifest",
                    f"expected {effective} plan rows, got {len(plan_rows)}",
                )

            atomic_write_jsonl(plan_path, sorted(plan_rows, key=lambda r: r["case_id"]))
            atomic_write_jsonl(
                self.config.manifest_dir / "_manifest.jsonl",
                [
                    {
                        "source_file_hash": source_hash,
                        "row_count": len(all_ids),
                        "corpus_rows_requested": requested,
                        "corpus_rows_effective": effective,
                        "created_at": utc_now(),
                    }
                ],
            )

            logger.info(
                "manifest built: %d cases (requested=%d, corpus=%d)",
                len(plan_rows),
                requested,
                len(all_ids),
            )
            return StageReport("manifest", len(plan_rows), 0, 0, time.monotonic() - started)

    # ════════════════════════════════════════════════════════════════
    # Stage: chunk — tokenizer-aware, resumable, deterministic
    # ════════════════════════════════════════════════════════════════

    def chunk(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()

            if self._is_halted("chunk") and not self.config.force:
                raise PipelineError("chunk", "halted")
            if self.config.force:
                self._clear_halted("chunk")

            self._check_inputs(include_metadata=False)
            plan = self._load_plan()
            if not plan:
                raise PipelineError("chunk", "plan is empty or missing")

            if self.config.dry_run:
                logger.info("chunk dry-run: %d cases", len(plan))
                return StageReport("chunk", 0, len(plan), 0, time.monotonic() - started)

            self.config.chunked_dir.mkdir(parents=True, exist_ok=True)
            previous = self._load_manifest(self.config.chunked_dir)
            failures = self._get_failures(self.config.chunked_dir)

            new_manifest: dict[int, dict] = {}
            processed = skipped = failed = 0

            logger.info(
                "chunk starting: %d cases, %d workers (chunker %s)",
                len(plan),
                self.config.workers,
                self.config.chunker_version,
            )

            tokenize = self._get_tokenizer() if self.config.workers == 1 else None

            executor: concurrent.futures.ProcessPoolExecutor | None = None
            if self.config.workers > 1:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.config.workers,
                    initializer=worker_init,
                    initargs=(str(self.config.model_dir),),
                )

            try:
                records_iter = self._iter_opinions(set(plan.keys()))
                pending: list[tuple[dict, concurrent.futures.Future]] = []

                def submit_next() -> bool:
                    nonlocal skipped
                    try:
                        record = next(records_iter)
                    except StopIteration:
                        return False
                    case_id = int(record["cl_opinion_id"])
                    plan_entry = plan[case_id]
                    if not self.config.force and self._is_valid_chunk_output(
                        case_id,
                        previous.get(case_id),
                        plan_entry,
                    ):
                        new_manifest[case_id] = previous[case_id]
                        skipped += 1
                        return True
                    if executor is None:
                        fut: concurrent.futures.Future = concurrent.futures.Future()
                        fut.set_result(chunk_worker(record, tokenize))
                    else:
                        fut = executor.submit(chunk_worker, record)
                    pending.append((record, fut))
                    return True

                max_pending = max(1, self.config.workers * 2)
                while len(pending) < max_pending and submit_next():
                    pass

                while pending:
                    record, fut = pending.pop(0)
                    try:
                        case_id, chunks, error, trace = fut.result()
                    except Exception as exc:
                        case_id = int(record["cl_opinion_id"])
                        chunks = None
                        error = f"{type(exc).__name__}: {exc}"
                        trace = traceback.format_exc()

                    processed += 1
                    if error is not None or chunks is None:
                        failed += 1
                        failures[case_id] = {
                            "case_id": case_id,
                            "error": error or "unknown",
                            "traceback": trace or "",
                            "stage": "chunk",
                            "ts": utc_now(),
                        }
                    else:
                        output_path = self._chunked_file(case_id)
                        atomic_write_jsonl(output_path, chunks)
                        new_manifest[case_id] = {
                            "case_id": case_id,
                            "source_text_hash": plan[case_id].source_text_hash,
                            "chunker_version": self.config.chunker_version,
                            "chunk_count": len(chunks),
                            "embedding_count": sum(1 for c in chunks if not c["is_parent"]),
                            "file_sha256": sha256_file(output_path),
                            "completed_at": utc_now(),
                        }
                        failures.pop(case_id, None)

                    if processed % FAILURE_CHECK_EVERY == 0 or processed == len(plan):
                        logger.info(
                            "chunk progress: %d/%d, skipped=%d, failed=%d",
                            processed,
                            len(plan),
                            skipped,
                            failed,
                        )

                    should_check = failed > FAILURE_ABSOLUTE or (
                        processed > 0 and processed % FAILURE_CHECK_EVERY == 0
                    )
                    if should_check:
                        self._persist_chunk_state(new_manifest, failures)
                        if self._check_failure_threshold(failed, processed):
                            reason = f"failure threshold exceeded: {failed}/{processed}"
                            self._write_halted("chunk", "systemic", failed, processed)
                            raise PipelineError(
                                "chunk",
                                "halted",
                                StageReport(
                                    "chunk",
                                    processed,
                                    skipped,
                                    failed,
                                    time.monotonic() - started,
                                    reason,
                                ),
                            )

                    while len(pending) < max_pending and submit_next():
                        pass

            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            self._persist_chunk_state(new_manifest, failures)
            if self._check_failure_threshold(failed, processed):
                reason = f"failure threshold exceeded: {failed}/{processed}"
                self._write_halted("chunk", "systemic", failed, processed)
                raise PipelineError(
                    "chunk",
                    "halted",
                    StageReport(
                        "chunk", processed, skipped, failed, time.monotonic() - started, reason
                    ),
                )

            orphans = self._cleanup_orphans(self.config.chunked_dir, set(new_manifest.keys()))
            if orphans:
                logger.info("chunk cleaned %d orphan files", orphans)

            logger.info(
                "chunk complete: %d processed, %d skipped, %d failed, %d orphans removed",
                processed,
                skipped,
                failed,
                orphans,
            )
            return StageReport("chunk", processed, skipped, failed, time.monotonic() - started)

    # ════════════════════════════════════════════════════════════════
    # Stage: embed — batched, per-case fallback, device-aware
    # ════════════════════════════════════════════════════════════════

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

            logger.info("embed starting: %d cases, device=%s", len(case_ids), self.config.device)
            embedder = self._get_embedder()
            logger.info("embed model loaded, beginning batched inference")

            for batch_start in range(0, len(case_ids), EMBED_CASES_PER_BATCH):
                case_batch = case_ids[batch_start : batch_start + EMBED_CASES_PER_BATCH]

                pending: list[tuple[int, dict, list[dict]]] = []
                for case_id in case_batch:
                    chunk_entry = chunk_manifest[case_id]
                    if not self.config.force and self._is_valid_embed_output(
                        case_id,
                        chunk_entry,
                        previous.get(case_id),
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
                    if processed and processed % FAILURE_CHECK_EVERY == 0:
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
                    for (case_id, chunk_entry, children), (slice_start, slice_end) in zip(
                        pending,
                        slices,
                        strict=True,
                    ):
                        try:
                            vectors = all_vecs[slice_start:slice_end]
                            if vectors.shape != (len(children), EMBEDDING_DIMENSIONS):
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
                            if per_vecs.shape != (len(children), EMBEDDING_DIMENSIONS):
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
                            "embed", processed, skipped, failed, time.monotonic() - started, reason
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
                        "embed", processed, skipped, failed, time.monotonic() - started, reason
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

    # ════════════════════════════════════════════════════════════════
    # Stage: load
    # ════════════════════════════════════════════════════════════════

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

        reserved = {"case_id", "cl_opinion_id", "cluster_id", "text", "text_kind", "metadata"}
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
            "precedential_status": choose("precedential_status"),
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
            raise PipelineError("load", f"orphan embeddings for case {case_id}: {len(vec_map)}")

    def load(self) -> StageReport:
        self._require_open()
        with self._stage_lock():
            started = time.monotonic()
            self._check_inputs(include_metadata=True)

            chunk_manifest = self._load_manifest(self.config.chunked_dir)
            embed_manifest = self._load_manifest(self.config.embedded_dir)

            if not chunk_manifest:
                raise PipelineError("load", "chunked manifest is empty")
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
                    raise PipelineError("load", f"invalid embed output for {case_id}")

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

            plan = self._load_plan()
            plan_ids = list(plan.keys())

            deleted = database.delete_stale_cases(plan_ids)
            if deleted:
                logger.info("deleted %d stale cases (cascades to edges and chunks)", deleted)

            clusters = self._load_clusters()

            def case_rows() -> Iterator[dict]:
                for record in self._iter_opinions(set(plan_ids)):
                    cid = record.get("cluster_id")
                    cluster = clusters.get(int(cid)) if cid is not None else None
                    yield self._join_cases(record, cluster)

            case_count = database.load_cases(case_rows())

            plan_ids_set = set(plan_ids)

            def edge_rows() -> Iterator[dict]:
                for row in self._iter_parquet(self.config.edges_file):
                    source = int(row["citing_opinion_id"])
                    target = int(row["cited_opinion_id"])
                    if source not in plan_ids_set or target not in plan_ids_set:
                        continue
                    yield {
                        "source_id": source,
                        "target_id": target,
                        "relation": "CITES",
                        "depth": int(row["depth"]) if row.get("depth") is not None else None,
                    }

            edge_count = database.load_edges(edge_rows())

            def chunk_rows() -> Iterator[dict]:
                for case_id in sorted(chunk_manifest):
                    yield from self._merge_chunks_and_vectors(case_id)

            staged = database.stage_chunks(chunk_rows())
            merged = database.merge_chunks()

            database.set_pipeline_state(
                PIPELINE_STATE_KEY,
                {
                    "dataset_fingerprint": fingerprint,
                    "chunker_version": self.config.chunker_version,
                    "embedder_version": self.config.embedder_version,
                    "case_count": case_count,
                    "edge_count": edge_count,
                    "staged_chunk_count": staged,
                    "merged_chunk_count": merged,
                    "loaded_at": utc_now(),
                },
            )

            counts = database.row_counts()
            logger.info("load complete: %s", json.dumps(counts))

            return StageReport("load", staged, 0, 0, time.monotonic() - started)

    # ════════════════════════════════════════════════════════════════
    # Stage: validate — 9 gates
    # ════════════════════════════════════════════════════════════════

    def _determinism_check(self, sample_size: int = 100) -> db.GateResult:
        chunk_manifest = self._load_manifest(self.config.chunked_dir)
        candidates = sorted(chunk_manifest)[:sample_size]
        if not candidates:
            return db.GateResult(
                name="gate_8_determinism",
                passed=False,
                value="no chunked opinions available",
                threshold="identical chunk IDs and hashes",
            )

        tokenize = self._get_tokenizer()
        sample = set(candidates)
        checked = 0
        mismatches: list[int] = []

        for record in self._iter_opinions(sample):
            case_id = int(record["cl_opinion_id"])
            if case_id not in sample:
                continue
            recomputed = chunker.chunk_opinion(record, tokenize=tokenize)
            expected = list(read_jsonl(self._chunked_file(case_id)))
            new_sig = [(c.chunk_id, c.text_hash, c.parent_id, c.is_parent) for c in recomputed]
            old_sig = [
                (r["chunk_id"], r["text_hash"], r.get("parent_id"), bool(r["is_parent"]))
                for r in expected
            ]
            checked += 1
            if new_sig != old_sig:
                mismatches.append(case_id)

        passed = checked == len(candidates) and not mismatches
        return db.GateResult(
            name="gate_8_determinism",
            passed=passed,
            value={"checked": checked, "expected": len(candidates), "mismatches": mismatches[:20]},
            threshold=f"{len(candidates)} identical chunk signatures",
        )

    def _embedding_determinism_check(self, sample_size: int = 10) -> db.GateResult:
        import numpy as np

        chunk_manifest = self._load_manifest(self.config.chunked_dir)
        candidates = sorted(chunk_manifest)[:sample_size]
        if not candidates:
            return db.GateResult(
                name="gate_9_embedding_determinism",
                passed=False,
                value="no chunked opinions",
                threshold="cosine >= 0.9999 on re-embed",
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
                    value=f"case {case_id}: {len(stored)} vectors != {len(children)} children",
                    threshold="cosine >= 0.9999",
                )
            new_vecs = embedder.embed_documents([r["text"] for r in children])
            old_vecs = np.asarray([r["vector"] for r in stored], dtype=np.float32)
            cosines = (new_vecs * old_vecs).sum(axis=1)
            worst = min(worst, float(cosines.min()))
            checked += 1

        return db.GateResult(
            name="gate_9_embedding_determinism",
            passed=worst >= 0.9999,
            value={"min_cosine": worst, "checked_cases": checked},
            threshold="cosine >= 0.9999",
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
                    StageReport("validate", 1, 0, len(failed), time.monotonic() - started),
                )
            return StageReport("validate", 1, 0, 0, time.monotonic() - started)

    # ════════════════════════════════════════════════════════════════
    # Stage: index
    # ════════════════════════════════════════════════════════════════

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

    # ════════════════════════════════════════════════════════════════
    # Stage: drop-index
    # ════════════════════════════════════════════════════════════════

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

    # ════════════════════════════════════════════════════════════════
    # Stage: run_all
    # ════════════════════════════════════════════════════════════════

    def run_all(self) -> list[StageReport]:
        self._require_open()
        if self.config.dry_run:
            return [
                StageReport(stage=name, processed=0, skipped=0, failed=0, duration_s=0.0)
                for name in ("manifest", "chunk", "embed", "load", "validate", "index")
            ]

        reports: list[StageReport] = []
        for stage in (self.manifest, self.chunk, self.embed, self.load, self.validate, self.index):
            reports.append(stage())
        return reports


# Re-exports for cli.py and other consumers — kept after the class definition
__all__ = [
    "IndexingPipeline",
    "Config",
    "PipelineError",
    "StageReport",
    "PlanEntry",
]
