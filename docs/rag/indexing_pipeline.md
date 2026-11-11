# Indexing Pipeline

## Overview
A modular, deterministic data ingestion pipeline designed to transform raw legal text corpora into a structured, vector-indexed PostgreSQL database. This project focuses on data integrity, reproducibility, and resource efficiency, implementing solid engineering practices to handle complex document processing reliably.

## Core Design Principles
- **Deterministic Execution**: Chunk IDs and file hashes are pure functions of the input data. The pipeline safely resumes from interruptions using SHA-256 validation and dataset fingerprinting.
- **Compute Efficiency**: Cross-case deduplication runs after chunking but before embedding, preventing wasted compute cycles on redundant text.
- **Crash-Safe I/O**: All file writes utilize an atomic temporary file replacement pattern (tempfile + fsync + rename) to prevent partial or corrupted writes during interruptions.
- **Strict Validation**: Eight SQL-based quality gates must pass before any database index is built, preventing malformed data from entering the retrieval system.
- **Separation of Concerns**: Chunking logic is isolated as a pure, I/O-free function, while database operations are confined to a dedicated module, enabling straightforward testing and parallelization.

## Pipeline Stages
The pipeline executes in a strict, dependency-ordered sequence. Each stage can be run independently or orchestrated together.

1. **Manifest**: Plans the ingestion run by hashing source Parquet files and resolving the target corpus size (min, medium, or max).
2. **Chunk**: Segments text into parent/child units using structure-aware splitting, atomic-span protection, and optional multiprocess execution.
3. **Dedup**: Removes cross-case duplicate children using exact SHA-256 hash matching, deterministically keeping the lowest case ID.
4. **Embed**: Generates vector embeddings via batched inference, supporting both strictly shaped OpenVINO (CPU) and PyTorch (CUDA) backends.
5. **Load**: Ingests data into PostgreSQL using temporary staging tables and safe ON CONFLICT upserts for cases, edges, and chunks.
6. **Validate**: Enforces rigorous data quality assertions against the staged data. Blocks progression if any gate fails.
7. **Index**: Builds concurrent HNSW (vector) and GIN (lexical) indexes only after validation succeeds.

## Validation Gates
Before the index stage is permitted to run, the validate stage executes the following SQL assertions. Failure results in an immediate pipeline halt and a structured error report.

1. **Parent Coverage**: 100% of child chunks must reference a valid, existing parent chunk.
2. **Breadcrumb Presence**: 100% of chunks must contain the required metadata header (case, citation, court, section).
3. **Boundary Integrity**: Less than or equal to 5% of child chunks may end mid-sentence.
4. **Citation Integrity**: Absolute zero tolerance for chunks ending in split citations.
5. **Size Distribution**: Enforces strict token limits (Child p50 <= 450, p95 <= 550, max <= 1600; Parent max <= 3000).
6. **Duplicate Content**: Zero intra-case duplicate child bodies permitted.
7. **Section Coverage**: Greater than or equal to 98% consistency between child and parent section metadata.
8. **Determinism**: Re-chunking a random sample of cases must yield byte-identical chunk IDs and hashes.

## File Structure
The pipeline is organized into focused modules to maintain clear boundaries between I/O, computation, and orchestration.

- `__init__.py` / `__main__.py`: Package initialization and entry points, allowing execution via `python -m indexing_pipeline`.
- `chunker.py`: Pure, deterministic text segmentation logic. Contains no I/O or database calls, ensuring it can be safely parallelized.
- `cli.py`: Command-line interface and argument parsing. Maps user flags to pipeline configuration.
- `db.py`: PostgreSQL schema management, connection pooling, and all SQL queries (upserts, validations, index creation).
- `dedup.py`: Cross-case exact-hash deduplication. Runs post-chunking to remove redundant text before embedding.
- `embedder.py`: Inference wrappers for both OpenVINO (CPU, statically shaped for memory safety) and PyTorch (CUDA, multi-GPU capable).
- `embedders/`: Directory containing the quantized OpenVINO model artifacts, tokenizer files, and compilation cache.
- `filters.py`: Metadata normalization and heuristic noise detection for short or malformed text.
- `io_utils.py`: Atomic file writing, SHA-256 hashing, and robust JSON/JSONL serialization utilities.
- `pipeline.py`: Main orchestration logic and ingest-side stages (manifest, chunk, dedup).
- `pipeline_helpers.py`: Configuration dataclasses, constants, worker initialization logic, and shared utility functions.
- `pipeline_load.py`: Load-side stages (embed, load, validate, index, drop-index), separated to avoid circular dependencies.
- `requirements.txt`: Pinned Python dependencies to ensure reproducible environments.

## Usage and CLI
The pipeline is orchestrated via a unified command-line interface. All stages share a common set of flags for consistency.

```bash
# Run the entire pipeline end-to-end
python -m indexing_pipeline all

# Run with maximum corpus size, 3 chunking workers, and JSON logging
python -m indexing_pipeline all --corpus-rows max --workers 3 --json-logs

# Force a full re-run, dropping existing indexes and ignoring skip logic
python -m indexing_pipeline all --force

# Run only the embedding stage
python -m indexing_pipeline embed

# Dry-run: Validate wiring and configuration without writing any data
python -m indexing_pipeline validate --dry-run
```

### Key Configuration Flags
- `--corpus-rows`: min (5,000), medium (100,000), max (400,000), or an explicit integer.
- `--device`: cpu (default, OpenVINO) or cuda / cuda:0,cuda:1 (PyTorch).
- `--no-dedup` / `--no-noise-filter`: Disable specific quality optimizations.
- `--db-dsn`: Override the DATABASE_URL environment variable.

## State and Fingerprinting
The pipeline tracks its state in the pipeline_state database table. A dataset fingerprint (SHA-256 hash) is computed from source file hashes, code versions (chunker, embedder, dedup, filters), and active backend configuration. If any of these change, the fingerprint changes. The load stage will detect this mismatch, refuse to skip, and force a complete re-ingestion, guaranteeing that the database never contains data generated by an outdated or incompatible pipeline version.
