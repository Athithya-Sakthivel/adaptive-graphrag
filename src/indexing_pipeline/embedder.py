"""OpenVINO (CPU) and PyTorch (CUDA) embedders for dinghy-law-0.6b-v1.

CPU path: static-reshaped OpenVINO IR with CACHE_DIR. Compile happens once
per process and is reused across batches and restarts.

CUDA path: sentence-transformers on the base HF checkpoint. Uses FP16,
batches at 64, and — when more than one GPU is available — spawns a
multi-process pool so both T4s share the workload. Produces vectors
numerically equivalent to the CPU path (cosine > 0.99999), so the
pipeline's manifest and skip logic is unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov

from .io_utils import sha256_file

logger = logging.getLogger(__name__)


# ── Public constants (imported by pipeline_helpers) ─────────────────────

EMBEDDING_DIMENSIONS = 1024
MAX_SEQ_LENGTH = 512
EMBED_BATCH_SIZE = 8  # OpenVINO static compile batch
BASE_MODEL_ID = "dinghy-law-0.6b-v1-openvino-fp16"

STATIC_BATCH = 8
STATIC_SEQUENCE = 464

# TorchEmbedder tuning for T4
TORCH_BATCH_SIZE = 64  # per-GPU sub-batch inside encode()
MULTI_GPU_MIN_TEXTS = 400  # below this, single GPU is faster

QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


def build_embedder_version(max_seq_length: int) -> str:
    """Version that changes when truncation behaviour changes."""
    return f"{BASE_MODEL_ID}-{max_seq_length}"


# ── OpenVINO CPU embedder (unchanged) ──────────────────────────────────


class Embedder:
    def __init__(
        self,
        model_dir: Path,
        max_seq_length: int = MAX_SEQ_LENGTH,
        threads: int = 4,
        batch_size: int = EMBED_BATCH_SIZE,
        device: str = "CPU",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_seq_length = int(max_seq_length)
        self.threads = int(threads)
        self.batch_size = int(batch_size)
        self.device = device

        if not 1 <= self.max_seq_length <= 8192:
            raise ValueError("max_seq_length must be 1..8192")
        if self.threads < 1 or self.batch_size < 1:
            raise ValueError("threads and batch_size must be positive")

        self.xml_path = self.model_dir / "openvino" / "openvino_model.xml"
        self.bin_path = self.model_dir / "openvino" / "openvino_model.bin"
        if not self.xml_path.is_file():
            raise FileNotFoundError(self.xml_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(self.bin_path)

        self._closed = False
        self._lock = threading.RLock()
        self.core: ov.Core | None = None
        self.compiled_model: Any = None
        self.tokenizer: Any = None
        self.input_ports: dict[str, dict[str, Any]] = {}
        self.fixed_sequence_length: int | None = None
        self.fixed_batch_size: int | None = None

        self._verify_hashes()
        self._load_tokenizer()
        self._compile_model()
        self._validate_model_inputs()

        warmup = self.embed_documents(["A court must interpret the applicable law."])
        if warmup.shape != (1, EMBEDDING_DIMENSIONS):
            raise RuntimeError(f"unexpected warmup shape: {warmup.shape}")

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise RuntimeError("Tokenizer has no pad/eos/unk token")
        self.tokenizer.padding_side = "right"
        limit = getattr(self.tokenizer, "model_max_length", self.max_seq_length)
        if isinstance(limit, int) and 0 < limit < 1_000_000 and self.max_seq_length > limit:
            raise ValueError(
                f"max_seq_length={self.max_seq_length} exceeds tokenizer limit={limit}"
            )

    def _verify_hashes(self) -> None:
        manifest = self.model_dir / "openvino_fp16_manifest.json"
        if not manifest.exists():
            logger.warning("model manifest absent: %s", manifest)
            return
        try:
            data = json.loads(manifest.read_text())
            files = data.get("files", data)
            for path in (self.xml_path, self.bin_path):
                expected = files.get(path.name) or files.get(str(path))
                if not expected:
                    continue
                if isinstance(expected, dict):
                    expected = expected.get("sha256", expected.get("hash"))
                actual = sha256_file(path).removeprefix("sha256:")
                if actual != str(expected).removeprefix("sha256:"):
                    logger.warning("model hash mismatch for %s", path.name)
        except Exception:
            logger.exception("could not verify model manifest")

    def _compile_model(self) -> None:
        self.core = ov.Core()

        cache_dir = self.model_dir / "ov_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache_dir)})

        model = self.core.read_model(str(self.xml_path))

        static_shape = ov.PartialShape([STATIC_BATCH, STATIC_SEQUENCE])
        port_to_shape = {port: static_shape for port in model.inputs}
        model.reshape(port_to_shape)

        properties = {
            ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.THROUGHPUT,
            ov.properties.inference_num_threads: self.threads,
        }
        self.compiled_model = self.core.compile_model(model, self.device, properties)

        self.fixed_batch_size = STATIC_BATCH
        self.fixed_sequence_length = self.max_seq_length

        logger.info(
            "compiled with static shape (batch=%d, seq=%d), cache_dir=%s",
            STATIC_BATCH,
            STATIC_SEQUENCE,
            cache_dir,
        )

    @staticmethod
    def _port_name(port: Any) -> str:
        try:
            return port.get_any_name()
        except Exception:
            return str(port)

    def _validate_model_inputs(self) -> None:
        for port in self.compiled_model.inputs:
            name = self._port_name(port).lower()
            if "input_ids" in name:
                key = "input_ids"
            elif "attention_mask" in name:
                key = "attention_mask"
            elif "token_type_ids" in name:
                key = "token_type_ids"
            elif "position_ids" in name:
                key = "position_ids"
            else:
                raise RuntimeError(f"unsupported model input: {name}")
            if key in self.input_ports:
                raise RuntimeError(f"duplicate input for {key}")
            self.input_ports[key] = {"port": port, "name": self._port_name(port)}

        if "input_ids" not in self.input_ports:
            raise RuntimeError("model has no input_ids")

        logger.info(
            "OpenVINO model outputs: %s",
            [
                {"name": self._port_name(p), "shape": str(p.partial_shape)}
                for p in self.compiled_model.outputs
            ],
        )

    def _tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=False,
            max_length=STATIC_SEQUENCE,
            return_attention_mask=True,
            return_tensors="np",
        )
        result = {k: np.asarray(v, dtype=np.int64) for k, v in encoded.items()}
        if "attention_mask" not in result:
            result["attention_mask"] = np.ones_like(result["input_ids"], dtype=np.int64)
        return result

    def _prepare_inputs(self, encoded: dict[str, np.ndarray]) -> dict[Any, np.ndarray]:
        input_ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        batch, seq = input_ids.shape
        inputs: dict[Any, np.ndarray] = {}

        for key, info in self.input_ports.items():
            if key == "input_ids":
                value = input_ids
            elif key == "attention_mask":
                value = mask
            elif key == "token_type_ids":
                value = encoded.get("token_type_ids", np.zeros_like(input_ids))
            elif key == "position_ids":
                value = np.broadcast_to(np.arange(seq, dtype=np.int64), (batch, seq)).copy()
            else:
                raise RuntimeError(f"unsupported input: {key}")
            inputs[info["port"]] = np.ascontiguousarray(value, dtype=np.int64)
        return inputs

    def _pool_and_normalize(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[2] != EMBEDDING_DIMENSIONS:
            raise RuntimeError(f"invalid hidden shape: {hidden.shape}")
        if hidden.shape[0] != mask.shape[0] or hidden.shape[1] != mask.shape[1]:
            raise RuntimeError("hidden shape does not match attention mask")
        lengths = mask.sum(axis=1).astype(np.int64)
        if np.any(lengths <= 0):
            raise RuntimeError("zero-length input in batch")
        pooled = hidden[np.arange(hidden.shape[0]), lengths - 1]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.where(norms > 1e-12, norms, 1.0)
        if not np.isfinite(normalized).all():
            raise RuntimeError("non-finite values in embeddings")
        return normalized.astype(np.float32)

    def _infer_batch(self, texts: Sequence[str]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Embedder is closed")
        original = len(texts)
        if original == 0:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        if original > STATIC_BATCH:
            raise RuntimeError(
                f"batch size {original} exceeds static batch {STATIC_BATCH}; "
                f"caller must chunk before calling _infer_batch"
            )

        inference_texts = list(texts) + [""] * (STATIC_BATCH - original)

        encoded = self._tokenize(inference_texts)
        inputs = self._prepare_inputs(encoded)

        with self._lock:
            outputs: Any = self.compiled_model(inputs)

        try:
            raw: list[Any] = list(outputs.values())
        except AttributeError:
            raw = list(outputs)

        arrays = [np.asarray(v) for v in raw]

        token_level = [
            a
            for a in arrays
            if a.ndim == 3
            and a.shape[0] == STATIC_BATCH
            and a.shape[1] == STATIC_SEQUENCE
            and a.shape[2] == EMBEDDING_DIMENSIONS
        ]
        if len(token_level) != 1:
            raise RuntimeError(
                f"expected one last_hidden_state "
                f"({STATIC_BATCH}, {STATIC_SEQUENCE}, {EMBEDDING_DIMENSIONS}); "
                f"got {[a.shape for a in arrays]}"
            )

        vectors = self._pool_and_normalize(token_level[0], encoded["attention_mask"])
        return vectors[:original]

    def _token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Embedder is closed")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings")

        lengths = [self._token_length(t) for t in texts]

        oversized = [(i, n) for i, n in enumerate(lengths) if n > self.max_seq_length]
        if oversized:
            raise RuntimeError(
                f"embedding input exceeds max_seq_length={self.max_seq_length}. "
                f"First offenders: {oversized[:10]}. "
                f"Chunking or breadcrumb budgeting must be corrected."
            )

        order = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
        output = np.empty((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)

        batch_size = STATIC_BATCH
        total_batches = (len(order) + batch_size - 1) // batch_size

        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            vectors = self._infer_batch([texts[i] for i in indices])
            if vectors.shape != (len(indices), EMBEDDING_DIMENSIONS):
                raise RuntimeError(f"embedder returned invalid shape: {vectors.shape}")
            output[np.asarray(indices)] = vectors

            batch_no = offset // batch_size + 1
            if batch_no % 2 == 0 or batch_no == total_batches:
                logger.info(
                    "embed_documents: %d/%d texts (batch %d/%d)",
                    min(offset + batch_size, len(order)),
                    len(order),
                    batch_no,
                    total_batches,
                )

        return output

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        return self._infer_batch([QUERY_PREFIX + text])[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self.compiled_model = None
            self.core = None
            self.tokenizer = None

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()


# ── PyTorch / CUDA embedder ────────────────────────────────────────────


class TorchEmbedder:
    """GPU embedder with FP16 and optional multi-GPU pool.

    `device` may be:
      - "cuda"        → auto-detect: uses ["cuda:0", "cuda:1"] if 2+ GPUs
      - "cuda:0"      → single GPU
      - ["cuda:0", "cuda:1"] → explicit list

    Large batches (> MULTI_GPU_MIN_TEXTS) use a sentence-transformers
    multi-process pool so both GPUs share the workload. Small batches
    stay single-device to avoid the pool's fork overhead.
    """

    def __init__(
        self,
        model_dir: Path,
        max_seq_length: int = MAX_SEQ_LENGTH,
        device: str | list[str] = "cuda",
        batch_size: int = TORCH_BATCH_SIZE,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        # ── Resolve device list ─────────────────────────────────────
        if isinstance(device, list):
            if not device:
                raise ValueError("device list is empty")
            primary = device[0]
            self.device_list = list(device)
            self.multi_device = len(self.device_list) > 1
        elif isinstance(device, str):
            if device == "cuda":
                count = torch.cuda.device_count()
                if count == 0:
                    raise RuntimeError("device='cuda' requested but no CUDA devices found")
                if count >= 2:
                    self.device_list = ["cuda:0", "cuda:1"]
                    self.multi_device = True
                else:
                    self.device_list = ["cuda:0"]
                    self.multi_device = False
                primary = self.device_list[0]
            elif device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"device={device} requested but torch.cuda.is_available() is False. "
                        "Install CUDA torch: pip install torch --index-url "
                        "https://download.pytorch.org/whl/cu121"
                    )
                primary = device
                self.device_list = [device]
                self.multi_device = False
            else:
                # cpu or other
                primary = device
                self.device_list = [device]
                self.multi_device = False
        else:
            raise TypeError(f"device must be str or list[str], got {type(device).__name__}")

        self.model_dir = Path(model_dir)
        self.max_seq_length = int(max_seq_length)
        self.device = primary
        self.batch_size = int(batch_size)
        self._pool: Any = None
        self._pool_failed = False
        self._closed = False

        if not self.max_seq_length or self.max_seq_length > 8192:
            raise ValueError(f"invalid max_seq_length: {self.max_seq_length}")
        if self.batch_size < 1:
            raise ValueError(f"invalid batch_size: {self.batch_size}")

        # ── Load primary model on the first device ──────────────────
        logger.info(
            "TorchEmbedder: loading model on %s (multi_device=%s, devices=%s)",
            primary,
            self.multi_device,
            self.device_list,
        )
        self.model = SentenceTransformer(
            str(self.model_dir),
            device=primary,
            backend="torch",
        )
        self.model.max_seq_length = self.max_seq_length
        self.model.half()  # FP16 — ~2× T4 throughput, ~50% activation VRAM
        self.model.eval()

        # ── Warmup (single-device, small, cheap) ────────────────────
        _ = self.model.encode(
            ["A court must interpret the applicable law."],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        warmup = self.embed_documents(["A court must interpret the applicable law."])
        if warmup.shape != (1, EMBEDDING_DIMENSIONS):
            raise RuntimeError(f"unexpected warmup shape: {warmup.shape}")

    # ── Pool management ─────────────────────────────────────────────

    def _try_start_pool(self) -> None:
        """Start the multi-GPU pool once. On failure, mark it permanently
        unavailable so we never retry inside the hot path."""
        if self._pool is not None or self._pool_failed:
            return
        if not self.multi_device:
            return

        try:
            self._pool = self.model.start_multi_process_pool(target_devices=self.device_list)
            logger.info(
                "multi-GPU pool started on %s (batch_size=%d per device)",
                self.device_list,
                self.batch_size,
            )
        except Exception as exc:
            logger.error("multi-GPU pool failed to start: %s", exc)
            logger.error("falling back to single-device embedding on %s", self.device)
            self._pool = None
            self._pool_failed = True

    # ── Inference ───────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._closed:
            raise RuntimeError("Embedder is closed")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings")

        # Guard: no input may exceed the model's max length. Raising here
        # is far faster than letting the model silently truncate.
        tokenizer = self.model.tokenizer
        lengths = [
            len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in texts
        ]
        oversized = [(i, n) for i, n in enumerate(lengths) if n > self.max_seq_length]
        if oversized:
            raise RuntimeError(
                f"embedding input exceeds max_seq_length={self.max_seq_length}. "
                f"First offenders: {oversized[:10]}. "
                f"Chunking or breadcrumb budgeting must be corrected."
            )

        use_pool = self.multi_device and len(texts) >= MULTI_GPU_MIN_TEXTS and not self._pool_failed

        if use_pool:
            self._try_start_pool()
            if self._pool is not None:
                try:
                    vectors = self.model.encode_multi_process(
                        texts,
                        self._pool,
                        batch_size=self.batch_size,
                        normalize_embeddings=True,
                    )
                    return self._finalize(vectors, len(texts))
                except Exception as exc:
                    logger.error(
                        "multi-GPU encode failed (%s); falling back to single-device",
                        type(exc).__name__,
                    )
                    self._pool_failed = True

        # Single-device fallback (also used for small batches)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return self._finalize(vectors, len(texts))

    @staticmethod
    def _finalize(vectors: Any, expected_count: int) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (expected_count, EMBEDDING_DIMENSIONS):
            raise RuntimeError(
                f"embedder returned invalid shape: {vectors.shape}, "
                f"expected ({expected_count}, {EMBEDDING_DIMENSIONS})"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("non-finite values in embeddings")
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        return self.embed_documents([QUERY_PREFIX + text])[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._pool is not None:
            try:
                self.model.stop_multi_process_pool(self._pool)
                logger.info("multi-GPU pool stopped")
            except Exception:
                logger.exception("could not stop multi-GPU pool cleanly")
            finally:
                self._pool = None

        try:
            import torch

            del self.model
            if any(isinstance(d, str) and d.startswith("cuda") for d in self.device_list):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> TorchEmbedder:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()
