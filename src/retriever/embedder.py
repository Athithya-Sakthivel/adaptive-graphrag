"""Query-only OpenVINO embedder for retrieval.

Compiled at (batch=1, sequence_length) with the LATENCY performance
hint for fast single-query inference on CPU. Completely independent of
``indexing_pipeline``'s batch-8 / GPU embedder.

This module must never import ``indexing_pipeline``. The separation is
load-bearing: the retriever process should not drag in the indexing
package's dependencies, and the two embedders have opposite profiles
(batch=1 vs batch=8, seq=128 vs seq=464, LATENCY vs THROUGHPUT,
CPU vs GPU).

Query prefix
------------
The instruction-tuned embedder expects a task prefix. Omitting it
degrades retrieval quality *silently* — the model still returns a
unit vector, just the wrong one. The prefix is applied inside
``embed_query`` so no caller can forget it.

No silent truncation
--------------------
Queries whose tokenized form (including the prefix and special
tokens) exceeds the compiled sequence length raise ``ValueError``.
Truncation would change retrieved results without warning; failing
loudly is the lesser evil and matches the router's guard contract.

Close semantics
---------------
``close()`` is idempotent. After close, ``embed_query`` raises
``RuntimeError`` rather than dereferencing a ``None`` compiled model.
All mutable state is declared ``X | None`` at class level so that
pyright flow-narrows correctly through the ``RLock`` guard.
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

# Expected hidden size of dinghy-law-0.6b. The class derives the true
# value from the compiled model at construction and only warns if it
# differs; the constant documents the contract downstream schema
# (halfvec(1024)) depends on.
EXPECTED_EMBEDDING_DIMENSIONS = 1024

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_THREADS = 4


# ── Path resolution ────────────────────────────────────────────────────


def _resolve_model_xml(model_dir: Path) -> Path:
    """Locate ``openvino_model.xml`` under either common layout."""
    candidates = (
        model_dir / "openvino" / "openvino_model.xml",
        model_dir / "openvino_model.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no openvino_model.xml under {model_dir}; "
        f"checked {[str(c.relative_to(model_dir)) for c in candidates]}"
    )


# ── Input port classification ──────────────────────────────────────────


def _classify_input_ports(ports: list[ov.Output]) -> dict[str, ov.Output]:
    """Map each input port to its semantic role.

    Recognised roles: ``input_ids``, ``attention_mask``,
    ``token_type_ids``, ``position_ids``.

    Every other port causes a hard error. Silently feeding the wrong
    tensor to an unfamiliar port produces a plausible-but-wrong
    vector — worse than failing at construction, when the cause is
    obvious.
    """
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
            f"recognised roles: {sorted(role_by_name)}"
        )

    missing = {"input_ids", "attention_mask"} - set(role_by_name)
    if missing:
        raise RuntimeError(f"missing required OpenVINO input ports: {sorted(missing)}")

    return role_by_name


# ── Output discovery (module-level; avoids self access before init) ───


def _discover_output(
    compiled: ov.CompiledModel,
    inputs: dict[str, ov.Output],
    sequence_length: int,
) -> tuple[ov.Output, int]:
    """Identify the 3D embedding output and its hidden dimension.

    Preferred path: a 3D output whose last dimension is already static
    after reshape (no inference needed).

    Fallback for a single dynamic-hidden output: run one dummy
    inference to read the true hidden size.
    """
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


# ── QueryEmbedder ──────────────────────────────────────────────────────


class QueryEmbedder:
    """Query-only embedder compiled for low-latency single-query CPU inference.

    Parameters
    ----------
    model_dir
        Directory containing the OpenVINO export and the HF tokenizer.
    sequence_length
        Static sequence length baked into the model graph. Queries
        whose tokenized form (including ``QUERY_PREFIX``) exceeds this
        raise ``ValueError`` rather than silently truncating.
    threads
        Value passed to ``ov.properties.inference_num_threads``. Set to
        the physical core count, not the logical count — hyperthreads
        contend for the same AVX2 execution ports and lengthen latency
        instead of shortening it.
    """

    # Always-set state (assigned in __init__, never cleared).
    model_dir: Path
    sequence_length: int
    threads: int
    max_seq_length: int
    tokenizer: PreTrainedTokenizerBase
    embedding_dimension: int

    # Optional state (set in __init__, cleared by close()).
    _core: ov.Core | None
    _compiled: ov.CompiledModel | None
    _inputs: dict[str, ov.Output]
    _output_port: ov.Output | None
    _lock: threading.RLock

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

        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer")
        if sequence_length < 16:
            raise ValueError(f"sequence_length must be >= 16, got {sequence_length}")

        if isinstance(threads, bool) or not isinstance(threads, int):
            raise TypeError("threads must be an integer")
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")

        self.sequence_length = int(sequence_length)
        self.threads = int(threads)
        # Alias used by the router's pre-flight guard.
        self.max_seq_length = self.sequence_length

        xml = _resolve_model_xml(self.model_dir)

        # ---- tokenizer (lazy import keeps transformers off the hot path) ----
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise RuntimeError(
                    f"tokenizer at {self.model_dir} has neither pad_token "
                    f"nor eos_token; cannot pad to max_length"
                )
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # ---- OpenVINO setup — all via locals so pyright sees non-Optional ----
        core = ov.Core()
        cache_dir = self.model_dir / "ov_cache_query"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})

        model = core.read_model(str(xml))
        model.reshape({port: ov.PartialShape([1, self.sequence_length]) for port in model.inputs})

        compiled = core.compile_model(
            model,
            "CPU",
            {
                ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY,
                ov.properties.inference_num_threads: self.threads,
            },
        )

        inputs = _classify_input_ports(list(compiled.inputs))
        output_port, embedding_dim = _discover_output(compiled, inputs, self.sequence_length)

        # ---- Assign to self atomically, after all setup succeeded ----
        self._core = core
        self._compiled = compiled
        self._inputs = inputs
        self._output_port = output_port
        self.embedding_dimension = embedding_dim
        self._lock = threading.RLock()

        if self.embedding_dimension != EXPECTED_EMBEDDING_DIMENSIONS:
            logger.warning(
                "model embedding dimension is %d, expected %d "
                "(downstream halfvec schema may reject this vector)",
                self.embedding_dimension,
                EXPECTED_EMBEDDING_DIMENSIONS,
            )

        # Warmup — surfaces tokenizer/model/pooling errors at
        # construction time, not at the first user query.
        self.embed_query("warmup")

    # ---- inference ----

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns an L2-normalised ``(D,)`` float32 vector.

        Raises ``ValueError`` if ``text`` is empty or its tokenized form
        (including ``QUERY_PREFIX``) exceeds ``self.sequence_length``.
        Truncation is never silent.
        """
        if not isinstance(text, str):
            raise TypeError(f"query must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("query must be non-empty")

        prefixed = QUERY_PREFIX + text

        # Deterministic pre-flight length check. ``encode()`` with
        # add_special_tokens=True returns the same real-token count
        # the batch call will place in input_ids[0, :L]. Doing this
        # before the padded call gives us the true token count in the
        # error message.
        tokens = self.tokenizer.encode(prefixed, add_special_tokens=True, truncation=False)
        token_count = len(tokens)
        if token_count > self.sequence_length:
            raise ValueError(
                f"query requires {token_count} tokens (including prefix "
                f"and special tokens); model permits {self.sequence_length}"
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

        # Grab locals under the lock so pyright sees non-Optional types
        # for the call. ``self._compiled`` and ``self._output_port`` are
        # mutable attributes; pyright does not model the RLock barrier
        # and cannot prove they stay non-None between the check and the
        # call.
        with self._lock:
            compiled = self._compiled
            output_port = self._output_port
            if compiled is None or output_port is None:
                raise RuntimeError("QueryEmbedder has been closed")
            raw_output = compiled(inputs)[output_port]

        hidden = np.asarray(raw_output)
        if hidden.shape != (1, self.sequence_length, self.embedding_dimension):
            raise RuntimeError(
                f"unexpected output shape {hidden.shape}; "
                f"expected (1, {self.sequence_length}, "
                f"{self.embedding_dimension})"
            )

        # Last-token pooling. With right-padding (the default, and what
        # ``padding="max_length"`` produces), the real tokens occupy
        # positions [0, real_len) so the last real token is at
        # real_len - 1.
        real_len = int(attention_mask[0].sum())
        if real_len < 1:
            raise RuntimeError("attention_mask sums to zero; cannot pool a real token")
        pooled = hidden[0, real_len - 1]  # (embedding_dimension,)

        if not np.isfinite(pooled).all():
            raise RuntimeError("model produced non-finite values in the last-token embedding")

        norm = float(np.linalg.norm(pooled))
        if norm < 1e-12:
            raise RuntimeError("model produced a zero vector; cannot L2-normalise")

        return (pooled / norm).astype(np.float32, copy=False)

    # ---- lifecycle ----

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
    "EXPECTED_EMBEDDING_DIMENSIONS",
]
