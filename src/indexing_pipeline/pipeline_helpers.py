"""Pipeline constants, config, errors, and worker functions.

Lower layer of the pipeline module. Imported by pipeline.py.
Does not import from pipeline.py — no circular dependency.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from . import chunker
from .embedder import MAX_SEQ_LENGTH, build_embedder_version

logger = logging.getLogger("indexing_pipeline")

# Suppress HF's "PyTorch was not found" banner.
logging.getLogger("transformers").setLevel(logging.ERROR)


# ── Derived paths (from __file__, no env vars) ──────────────────────────

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent

DEFAULT_DATA_DIR = _REPO_ROOT / "tmp" / "data"
DEFAULT_MODEL_DIR = _PKG_DIR / "embedders" / "dinghy-law-0.6b-v1-openvino-fp16"
DEFAULT_GPU_MODEL_DIR = _PKG_DIR / "models" / "dinghy-law-0.6b-v1"


# ── Corpus-size presets ─────────────────────────────────────────────────

PRESETS: dict[str, int] = {
    "min": 5_000,
    "medium": 100_000,
    "max": 400_000,
}
DEFAULT_CORPUS_ROWS = "min"


def resolve_corpus_rows(value: str) -> int:
    v = str(value).strip().lower()
    if v in PRESETS:
        return PRESETS[v]
    if v.isdigit():
        n = int(v)
        if n < 1:
            raise ValueError(f"CORPUS_ROWS must be positive, got {value!r}")
        return n
    raise ValueError(
        f"CORPUS_ROWS must be 'min', 'medium', 'max', or a positive integer; got {value!r}"
    )


# ── Tunables (CLI flags can override) ───────────────────────────────────

WORKERS = 1
BATCH_SIZE = 500
EMBED_BATCH = 4
EMBED_CASES_PER_BATCH = 32
INFERENCE_THREADS = 4
POOL_MIN = 1
POOL_MAX = 4
DEVICE = "cpu"
WORKER_MODE_MAX = 3

FAILURE_ABSOLUTE = 20
FAILURE_RATE = 0.05
FAILURE_CHECK_EVERY = 10

PIPELINE_STATE_KEY = "bulk_indexing_pipeline"

CORPUS_REL = Path("raw/corpus/opinions.parquet")
CLUSTERS_REL = Path("metadata/clusters_filtered.parquet")
EDGES_REL = Path("graph/citations_filtered.parquet")


# ── Errors and reports ──────────────────────────────────────────────────


class PipelineError(Exception):
    def __init__(self, stage: str, reason: str, report: StageReport | None = None) -> None:
        self.stage = stage
        self.reason = reason
        self.report = report
        super().__init__(f"stage={stage}: {reason}")


@dataclass(frozen=True, slots=True)
class StageReport:
    stage: str
    processed: int
    skipped: int
    failed: int
    duration_s: float
    halt_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlanEntry:
    case_id: int
    source_text_hash: str
    text_kind: str
    cluster_id: int | None


# ── Config ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Config:
    data_dir: Path
    model_dir: Path
    db_dsn: str
    gpu_model_dir: Path = DEFAULT_GPU_MODEL_DIR

    max_seq_length: int = MAX_SEQ_LENGTH
    corpus_rows: str = DEFAULT_CORPUS_ROWS
    workers: int = WORKERS
    batch_size: int = BATCH_SIZE
    embed_batch: int = EMBED_BATCH
    inference_threads: int = INFERENCE_THREADS
    pool_min: int = POOL_MIN
    pool_max: int = POOL_MAX
    device: str = DEVICE

    dry_run: bool = False
    force: bool = False

    @property
    def manifest_dir(self) -> Path:
        return self.data_dir / "manifest"

    @property
    def chunked_dir(self) -> Path:
        return self.data_dir / "chunked"

    @property
    def embedded_dir(self) -> Path:
        return self.data_dir / "embedded"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def corpus_file(self) -> Path:
        return self.data_dir / CORPUS_REL

    @property
    def clusters_file(self) -> Path:
        return self.data_dir / CLUSTERS_REL

    @property
    def edges_file(self) -> Path:
        return self.data_dir / EDGES_REL

    @property
    def chunker_version(self) -> str:
        return chunker.CHUNKER_VERSION

    @property
    def embedder_version(self) -> str:
        # Device-independent: GPU and CPU produce equivalent vectors.
        return build_embedder_version(self.max_seq_length)

    def validated(self) -> Config:
        if not isinstance(self.data_dir, Path):
            raise TypeError("data_dir must be a Path")
        if not isinstance(self.model_dir, Path):
            raise TypeError("model_dir must be a Path")
        if not isinstance(self.gpu_model_dir, Path):
            raise TypeError("gpu_model_dir must be a Path")
        # db_dsn may be empty — required only for load/validate/index/drop-index.
        if self.db_dsn and not self.db_dsn.strip():
            raise ValueError("db_dsn, when provided, must be non-empty after strip")
        if not 1 <= self.max_seq_length <= 8192:
            raise ValueError(f"max_seq_length must be 1..8192, got {self.max_seq_length}")
        if not 1 <= self.workers <= WORKER_MODE_MAX:
            raise ValueError(f"workers must be 1..{WORKER_MODE_MAX}, got {self.workers}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.embed_batch < 1:
            raise ValueError(f"embed_batch must be >= 1, got {self.embed_batch}")
        if self.inference_threads < 1:
            raise ValueError(f"inference_threads must be >= 1, got {self.inference_threads}")
        if self.pool_min < 1 or self.pool_max < self.pool_min:
            raise ValueError(f"invalid pool: min={self.pool_min}, max={self.pool_max}")
        if not self.device or not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda[:N]'; got {self.device!r}")
        resolve_corpus_rows(self.corpus_rows)
        return self

    @classmethod
    def from_env(cls) -> Config:
        # DATABASE_URL is optional. Stages that need Postgres (load, validate,
        # index, drop-index) raise a clear PipelineError when db_dsn is empty.
        # The embed stage never opens a database connection, so it runs without.
        dsn = os.environ.get("DATABASE_URL", "").strip()
        return cls(
            data_dir=DEFAULT_DATA_DIR,
            model_dir=DEFAULT_MODEL_DIR,
            gpu_model_dir=DEFAULT_GPU_MODEL_DIR,
            db_dsn=dsn,
            corpus_rows=os.environ.get("CORPUS_ROWS", DEFAULT_CORPUS_ROWS),
            device=os.environ.get("DEVICE", DEVICE),
        ).validated()

    def with_overrides(self, **kwargs: Any) -> Config:
        allowed = {
            "data_dir",
            "model_dir",
            "gpu_model_dir",
            "db_dsn",
            "max_seq_length",
            "corpus_rows",
            "workers",
            "batch_size",
            "embed_batch",
            "inference_threads",
            "pool_min",
            "pool_max",
            "device",
            "dry_run",
            "force",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"unsupported override keys: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in ("data_dir", "model_dir", "gpu_model_dir"):
                values[key] = Path(value).expanduser().resolve()
            else:
                values[key] = value
        return replace(self, **values).validated()


# ── Worker functions (top-level for ProcessPoolExecutor picklability) ───

# Populated in each child process by worker_init.
_WORKER_TOKENIZE: Callable[[str], int] | None = None


def _make_tokenize(path_str: str) -> Callable[[str], int]:
    """Return a closure that tokenizes with the tokenizer loaded from path_str.

    The closure binds `tok` at call time of _make_tokenize, so the returned
    function is independent of any loop variable in the caller.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path_str, local_files_only=True)

    def tokenize(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=True, truncation=False))

    return tokenize


def worker_init(model_dir_str: str, gpu_model_dir_str: str = "") -> None:
    """Load the embedder's tokenizer once per worker process.

    Tries model_dir first (CPU OpenVINO IR directory), then gpu_model_dir
    (base HF checkpoint). On Kaggle only the GPU directory exists: model_dir
    is the OpenVINO IR path, which is not downloaded there.
    """
    global _WORKER_TOKENIZE

    candidates = [p for p in (model_dir_str, gpu_model_dir_str) if p]
    for path_str in candidates:
        try:
            _WORKER_TOKENIZE = _make_tokenize(path_str)
            logger.info("worker tokenizer loaded from %s", path_str)
            return
        except Exception:
            continue

    logger.error(
        "worker failed to load tokenizer from any of %s; falling back to chars/4 heuristic",
        candidates,
    )
    _WORKER_TOKENIZE = None


def chunk_worker(
    record: dict,
    tokenize: Callable[[str], int] | None = None,
) -> tuple[int, list[dict] | None, str | None, str | None]:
    """One record in, chunks list out. Runs in main process or worker pool."""
    global _WORKER_TOKENIZE

    # Prefer the explicitly-passed tokenizer (main-process path).
    # Fall back to the worker-loaded one (ProcessPoolExecutor path).
    measure = tokenize if tokenize is not None else _WORKER_TOKENIZE

    raw_id = record.get("cl_opinion_id")
    if raw_id is None:
        return -1, None, "missing cl_opinion_id", ""
    try:
        case_id = int(raw_id)
    except (TypeError, ValueError, OverflowError):
        return -1, None, f"invalid cl_opinion_id: {raw_id!r}", ""

    try:
        chunks = chunker.chunk_opinion(record, tokenize=measure)
        return case_id, [asdict(c) for c in chunks], None, None
    except Exception as exc:
        return case_id, None, f"{type(exc).__name__}: {exc}", traceback.format_exc()
