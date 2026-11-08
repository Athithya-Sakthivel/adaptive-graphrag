"""OpenVINO (CPU) and PyTorch (CUDA) embedders for the legal RAG pipeline.

CPU Path Notes:
- Uses a strict static-reshaped OpenVINO IR (batch=32, seq=512) to prevent
  memory spikes (OOM) while maintaining maximum CPU throughput.
- Tokenization is batched using the fast C++ tokenizer to avoid Python loops.
- Removed the slow "sort by length" logic which provided zero benefit for
  static shapes and added massive latency overhead.
- Relies on OpenVINO's auto-tuning THROUGHPUT mode.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 384
MAX_SEQ_LENGTH = 512

# Memory-safe defaults for a 16GB laptop.
# batch=32 prevents OOM; threads=8 matches physical cores (ignores SMT).
EMBED_BATCH_SIZE = 32
DEFAULT_THREADS = 8
BASE_MODEL_ID = "bge-small-en-v1.5-openvino-qint8"

QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


def build_embedder_version(max_seq_length: int) -> str:
    """Version that changes when truncation behaviour changes."""
    return f"{BASE_MODEL_ID}-{max_seq_length}"


# ── OpenVINO CPU Embedder ────────────────────────────────────────────────


class Embedder:
    def __init__(
        self,
        model_dir: Path,
        max_seq_length: int = MAX_SEQ_LENGTH,
        threads: int = DEFAULT_THREADS,
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

        self.xml_path = self.model_dir / "openvino_model_qint8_quantized.xml"
        self.bin_path = self.model_dir / "openvino_model_qint8_quantized.bin"
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Missing OpenVINO XML: {self.xml_path}")
        if not self.bin_path.is_file():
            raise FileNotFoundError(f"Missing OpenVINO BIN: {self.bin_path}")

        self._closed = False
        self._lock = threading.RLock()
        self.core: ov.Core | None = None
        self.compiled_model: Any = None
        self.tokenizer: Any = None
        self.input_ports: dict[str, dict[str, Any]] = {}

        self._load_tokenizer()
        self._compile_model()
        self._validate_model_inputs()

        # Warmup
        warmup = self.embed_documents(["A court must interpret the applicable law."])
        if warmup.shape != (1, EMBEDDING_DIMENSIONS):
            raise RuntimeError(f"Unexpected warmup shape: {warmup.shape}")

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _compile_model(self) -> None:
        self.core = ov.Core()
        cache_dir = self.model_dir / "ov_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache_dir)})

        model = self.core.read_model(str(self.xml_path))

        # Strict static shape to maximize CPU throughput and prevent OOM
        static_shape = ov.PartialShape([self.batch_size, self.max_seq_length])
        model.reshape({port: static_shape for port in model.inputs})

        # OpenVINO auto-tunes streams based on THROUGHPUT hint
        properties = {
            ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.THROUGHPUT,
            ov.properties.inference_num_threads: self.threads,
        }

        self.compiled_model = self.core.compile_model(model, self.device, properties)
        logger.info(
            "OpenVINO compiled: batch=%d, seq=%d, threads=%d (Auto-THROUGHPUT)",
            self.batch_size,
            self.max_seq_length,
            self.threads,
        )

    def _validate_model_inputs(self) -> None:
        for port in self.compiled_model.inputs:
            name = port.get_any_name().lower()
            if "input_ids" in name:
                key = "input_ids"
            elif "attention_mask" in name:
                key = "attention_mask"
            elif "token_type_ids" in name:
                key = "token_type_ids"
            else:
                continue
            self.input_ports[key] = {"port": port}

    def _infer_batch(self, texts: Sequence[str]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Embedder is closed")
        original = len(texts)
        if original == 0:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)

        # Pad to static batch size
        inference_texts = list(texts) + [""] * (self.batch_size - original)

        # FAST C++ TOKENIZATION (No Python loops)
        encoded = self.tokenizer(
            inference_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
            return_tensors="np",
        )

        inputs = {}
        for key, info in self.input_ports.items():
            if key == "input_ids":
                inputs[info["port"]] = np.ascontiguousarray(encoded["input_ids"], dtype=np.int64)
            elif key == "attention_mask":
                inputs[info["port"]] = np.ascontiguousarray(
                    encoded["attention_mask"], dtype=np.int64
                )
            elif key == "token_type_ids":
                inputs[info["port"]] = np.zeros_like(encoded["input_ids"], dtype=np.int64)

        with self._lock:
            outputs = self.compiled_model(inputs)

        raw = list(outputs.values()) if hasattr(outputs, "values") else list(outputs)
        hidden = np.asarray(raw[0], dtype=np.float32)

        # Pooling (CLS token is index 0 for BGE models)
        pooled = hidden[np.arange(self.batch_size), 0]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.where(norms > 1e-12, norms, 1.0)

        return normalized[:original]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Embedder is closed")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)

        output = np.empty((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for offset in range(0, len(texts), self.batch_size):
            batch_texts = texts[offset : offset + self.batch_size]
            vectors = self._infer_batch(batch_texts)
            output[offset : offset + len(batch_texts)] = vectors

            batch_no = offset // self.batch_size + 1
            if batch_no % 2 == 0 or batch_no == total_batches:
                logger.info(
                    "embed_documents: %d/%d texts (batch %d/%d)",
                    min(offset + self.batch_size, len(texts)),
                    len(texts),
                    batch_no,
                    total_batches,
                )

        return output

    def embed_query(self, text: str) -> np.ndarray:
        return self._infer_batch([QUERY_PREFIX + text])[0]

    def close(self) -> None:
        self._closed = True
        self.compiled_model = None
        self.core = None
        self.tokenizer = None

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()


# ── PyTorch / CUDA Embedder ──────────────────────────────────────────────
# (TorchEmbedder stays as it was, untouched by CPU performance tuning)
class TorchEmbedder:
    def __init__(
        self,
        model_dir: Path,
        max_seq_length: int = MAX_SEQ_LENGTH,
        device: str | list[str] = "cuda",
        batch_size: int = 64,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if isinstance(device, list):
            if not device:
                raise ValueError("Device list is empty")
            primary = device[0]
            self.device_list = list(device)
            self.multi_device = len(self.device_list) > 1
        elif isinstance(device, str):
            if device == "cuda":
                count = torch.cuda.device_count()
                if count == 0:
                    raise RuntimeError("Device='cuda' requested but no CUDA devices found")
                if count >= 2:
                    self.device_list = ["cuda:0", "cuda:1"]
                    self.multi_device = True
                else:
                    self.device_list = ["cuda:0"]
                    self.multi_device = False
                primary = self.device_list[0]
            elif device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(f"Device={device} requested but CUDA is unavailable.")
                primary = device
                self.device_list = [device]
                self.multi_device = False
            else:
                primary = device
                self.device_list = [device]
                self.multi_device = False
        else:
            raise TypeError(f"Device must be str or list[str], got {type(device).__name__}")

        self.model_dir = Path(model_dir)
        self.max_seq_length = int(max_seq_length)
        self.device = primary
        self.batch_size = int(batch_size)
        self._pool: Any = None
        self._pool_failed = False
        self._closed = False

        self.model = SentenceTransformer(str(self.model_dir), device=primary, backend="torch")
        self.model.max_seq_length = self.max_seq_length
        self.model.half()
        self.model.eval()

        _ = self.model.encode(
            ["A court must interpret the applicable law."], batch_size=1, normalize_embeddings=True
        )
        if self.device.startswith("cuda") and self._pool is None:
            torch.cuda.empty_cache()

    def _try_start_pool(self) -> None:
        if self._pool is not None or self._pool_failed or not self.multi_device:
            return
        try:
            self._pool = self.model.start_multi_process_pool(target_devices=self.device_list)
        except Exception as exc:
            logger.error("Multi-GPU pool failed: %s", exc)
            self._pool_failed = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._closed:
            raise RuntimeError("Embedder is closed")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All inputs must be strings")

        tokenizer = self.model.tokenizer
        lengths = [
            len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in texts
        ]
        if any(n > self.max_seq_length for n in lengths):
            raise RuntimeError(f"Input exceeds max_seq_length={self.max_seq_length}.")

        use_pool = self.multi_device and len(texts) >= 20_000 and not self._pool_failed

        if use_pool:
            self._try_start_pool()
            if self._pool is not None:
                try:
                    vectors = self.model.encode_multi_process(
                        texts, self._pool, batch_size=self.batch_size, normalize_embeddings=True
                    )
                    return self._finalize(vectors, len(texts))
                except Exception:
                    self._pool_failed = True

        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        if self.device.startswith("cuda") and self._pool is None:
            torch.cuda.empty_cache()

        return self._finalize(vectors, len(texts))

    @staticmethod
    def _finalize(vectors: Any, expected_count: int) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (expected_count, EMBEDDING_DIMENSIONS):
            raise RuntimeError(f"Embedder returned invalid shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError("Non-finite values in embeddings")
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a non-empty string")
        return self.embed_documents([QUERY_PREFIX + text])[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            try:
                self.model.stop_multi_process_pool(self._pool)
            except Exception:
                pass
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
