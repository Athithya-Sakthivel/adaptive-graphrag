"""Query-only OpenVINO embedder for retrieval.

Compiled at (batch=1, sequence_length) with LATENCY performance hint
for fast single-query inference on CPU. Completely independent of the
indexing pipeline's batch embedder.

Query prefix: instruction-tuned embedders (e.g., BGE-instruct) expect
a task prefix. Omitting it degrades retrieval quality silently — the
model still returns a unit vector, just the wrong one.

No silent truncation: queries whose tokenized form exceeds the
compiled sequence length raise ValueError rather than truncating.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import openvino as ov

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

EMBEDDING_DIMENSIONS = 384
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_THREADS = 4


def _resolve_model_xml(model_dir: Path) -> Path:
    """Locate OpenVINO model XML under common layouts."""
    candidates = (
        model_dir / "openvino_model_qint8_quantized.xml",
        model_dir / "openvino" / "openvino_model.xml",
        model_dir / "openvino_model.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no OpenVINO model XML found under {model_dir}; "
        f"checked {[str(c.relative_to(model_dir)) for c in candidates]}"
    )


def _classify_input_ports(ports: list[ov.Output]) -> dict[str, ov.Output]:
    """Map each input port to its semantic role (input_ids, attention_mask, etc.)."""
    role_by_name: dict[str, ov.Output] = {}
    unexpected: list[str] = []

    for port in ports:
        try:
            name = port.get_any_name().lower()
        except RuntimeError:
            unexpected.append("<unnamed>")
            continue

        if "input_ids" in name:
            role_by_name["input_ids"] = port
        elif "attention_mask" in name:
            role_by_name["attention_mask"] = port
        elif "token_type_ids" in name:
            role_by_name["token_type_ids"] = port
        elif "position_ids" in name:
            role_by_name["position_ids"] = port
        else:
            unexpected.append(name)

    if unexpected:
        raise RuntimeError(
            f"unexpected OpenVINO input ports: {unexpected}; "
            f"recognized roles: {sorted(role_by_name)}"
        )

    missing = {"input_ids", "attention_mask"} - set(role_by_name)
    if missing:
        raise RuntimeError(f"missing required OpenVINO input ports: {sorted(missing)}")

    return role_by_name


def _discover_output(
    compiled: ov.CompiledModel,
    inputs: dict[str, ov.Output],
    sequence_length: int,
) -> tuple[ov.Output, int]:
    """Identify the 3D embedding output and its hidden dimension."""
    outputs = list(compiled.outputs)
    if not outputs:
        raise RuntimeError("compiled model has no outputs")

    for port in outputs:
        shape = port.partial_shape
        if len(shape) == 3 and shape[2].is_static:
            return port, int(shape[2].get_length())

    if len(outputs) == 1:
        dummy_ids = np.zeros((1, sequence_length), dtype=np.int64)
        dummy_mask = np.zeros((1, sequence_length), dtype=np.int64)
        dummy_mask[0, 0] = 1
        feed: dict[Any, np.ndarray] = {
            inputs["input_ids"]: dummy_ids,
            inputs["attention_mask"]: dummy_mask,
        }
        if "token_type_ids" in inputs:
            feed[inputs["token_type_ids"]] = dummy_ids
        if "position_ids" in inputs:
            feed[inputs["position_ids"]] = np.arange(sequence_length, dtype=np.int64)[None, :]
        result = np.asarray(compiled(feed)[outputs[0]])
        if result.ndim != 3:
            raise RuntimeError(f"single output has shape {result.shape}, expected 3D")
        return outputs[0], int(result.shape[2])

    raise RuntimeError(
        f"cannot identify embedding output among {len(outputs)} outputs: "
        f"{[str(p.partial_shape) for p in outputs]}"
    )


class QueryEmbedder:
    """Query-only embedder compiled for low-latency single-query CPU inference.

    Args:
        model_dir: Directory containing OpenVINO export and HF tokenizer.
        sequence_length: Static sequence length baked into model graph.
            Queries exceeding this raise ValueError.
        threads: Value passed to ov.properties.inference_num_threads.
            Set to physical core count (not logical).
    """

    def __init__(
        self,
        model_dir: Path | str,
        *,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model_dir does not exist: {self.model_dir}")

        if not isinstance(sequence_length, int) or sequence_length < 16:
            raise ValueError(f"sequence_length must be int >= 16, got {sequence_length}")
        if not isinstance(threads, int) or threads < 1:
            raise ValueError(f"threads must be int >= 1, got {threads}")

        self.sequence_length = sequence_length
        self.threads = threads
        self.max_seq_length = sequence_length

        xml = _resolve_model_xml(self.model_dir)

        from transformers import AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise RuntimeError(
                    f"tokenizer at {self.model_dir} has neither pad_token nor eos_token"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._core = ov.Core()
        cache_dir = self.model_dir / "ov_cache_query"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._core.set_property({"CACHE_DIR": str(cache_dir)})

        model = self._core.read_model(str(xml))
        model.reshape({port: ov.PartialShape([1, self.sequence_length]) for port in model.inputs})

        self._compiled = self._core.compile_model(
            model,
            "CPU",
            {
                ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY,
                ov.properties.inference_num_threads: self.threads,
            },
        )

        self._inputs = _classify_input_ports(list(self._compiled.inputs))
        self._output_port, self.embedding_dimension = _discover_output(
            self._compiled, self._inputs, self.sequence_length
        )
        self._lock = threading.RLock()

        if self.embedding_dimension != EMBEDDING_DIMENSIONS:
            logger.warning(
                "model embedding dim is %d, expected %d (downstream halfvec schema may reject)",
                self.embedding_dimension,
                EMBEDDING_DIMENSIONS,
            )

        self.embed_query("warmup")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns L2-normalized (D,) float32 vector.

        Raises ValueError if text is empty or tokenized form exceeds
        the compiled sequence length. Truncation is never silent.
        """
        if not isinstance(text, str):
            raise TypeError(f"query must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("query must be non-empty")

        prefixed = QUERY_PREFIX + text

        tokens = self.tokenizer.encode(prefixed, add_special_tokens=True, truncation=False)
        if len(tokens) > self.sequence_length:
            raise ValueError(
                f"query requires {len(tokens)} tokens (including prefix and special tokens); "
                f"model permits {self.sequence_length}"
            )

        encoded = self.tokenizer(
            [prefixed],
            add_special_tokens=True,
            padding="max_length",
            truncation=False,
            max_length=self.sequence_length,
            return_attention_mask=True,
            return_tensors="np",
        )

        input_ids = np.ascontiguousarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.ascontiguousarray(encoded["attention_mask"], dtype=np.int64)

        inputs: dict[Any, np.ndarray] = {
            self._inputs["input_ids"]: input_ids,
            self._inputs["attention_mask"]: attention_mask,
        }

        if "token_type_ids" in self._inputs:
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = np.zeros_like(input_ids)
            inputs[self._inputs["token_type_ids"]] = np.ascontiguousarray(
                token_type_ids, dtype=np.int64
            )

        if "position_ids" in self._inputs:
            inputs[self._inputs["position_ids"]] = np.ascontiguousarray(
                np.arange(self.sequence_length, dtype=np.int64)[None, :]
            )

        with self._lock:
            if self._compiled is None or self._output_port is None:
                raise RuntimeError("QueryEmbedder has been closed")
            raw_output = self._compiled(inputs)[self._output_port]

        hidden = np.asarray(raw_output)
        expected_shape = (1, self.sequence_length, self.embedding_dimension)
        if hidden.shape != expected_shape:
            raise RuntimeError(f"unexpected output shape {hidden.shape}; expected {expected_shape}")

        real_len = int(attention_mask[0].sum())
        if real_len < 1:
            raise RuntimeError("attention_mask sums to zero; cannot pool")

        pooled = hidden[0, 0]

        if not np.isfinite(pooled).all():
            raise RuntimeError("model produced non-finite values in CLS embedding")

        norm = float(np.linalg.norm(pooled))
        if norm < 1e-12:
            raise RuntimeError("model produced zero vector; cannot L2-normalize")

        return (pooled / norm).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release OpenVINO resources. Idempotent and thread-safe."""
        with self._lock:
            self._compiled = None
            self._inputs = {}
            self._output_port = None
            self._core = None

    def __enter__(self) -> QueryEmbedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"QueryEmbedder(model_dir={self.model_dir.name!r}, "
            f"seq={self.sequence_length}, threads={self.threads}, "
            f"dim={self.embedding_dimension})"
        )


__all__ = [
    "QueryEmbedder",
    "QUERY_PREFIX",
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_THREADS",
    "EMBEDDING_DIMENSIONS",
]
