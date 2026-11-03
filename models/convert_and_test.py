#!/usr/bin/env python3
"""Export dinghy-law-0.6b-v1 to OpenVINO FP16 IR and validate it on CPU.

Run:
    python convert_and_test.py

Outputs:
    dinghy_law_openvino_fp16/   Sentence-Transformers model + OpenVINO XML/BIN
    openvino_fp16_validation.json

This script creates FP16 *stored weights* in OpenVINO IR. It does not perform
INT8 quantization and does not create or execute an ONNX artifact.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------- User configuration ---------------------------
MODEL_ID = "Hanno-Labs/dinghy-law-0.6b-v1"
# Optional: set MODEL_REVISION to a tag, branch, or immutable Hugging Face commit.
MODEL_REVISION = os.getenv("MODEL_REVISION") or None
MAX_SEQ_LENGTH = 8192
EXPECTED_EMBEDDING_DIM = 1024
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "2")))
CPU_THREADS = max(1, int(os.getenv("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))))

# FP16 is not lossless. These are acceptance thresholds against the FP32
# Sentence-Transformers reference and can be tightened/relaxed deliberately.
MIN_COSINE = 0.9990
MIN_MEAN_COSINE = 0.9995
MIN_TOP5_JACCARD = 0.90
MAX_MRR_DROP = 0.01
TOP_K = 5
TIMING_REPEATS = 2
OV_PERFORMANCE_HINT = os.getenv("OV_PERFORMANCE_HINT", "LATENCY").upper()
if OV_PERFORMANCE_HINT not in {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
    raise ValueError("OV_PERFORMANCE_HINT must be LATENCY, THROUGHPUT, or CUMULATIVE_THROUGHPUT")

BASE = Path(__file__).resolve().parent
OPENVINO_DIR = BASE / "dinghy_law_openvino_fp16"
SUMMARY_FILE = BASE / "openvino_fp16_validation.json"
MANIFEST_NAME = "openvino_fp16_manifest.json"

# Small deterministic smoke-test set. This is not a legal-retrieval benchmark.
QUERIES = [
    "When does the limitation period for a personal injury claim begin?",
    "Can fraudulent concealment justify equitable tolling of a filing deadline?",
    "What is the legal standard for granting summary judgment?",
    "What process is required before the government deprives a person of property?",
]
DOCUMENTS = [
    "For a personal injury claim, the statute of limitations generally begins to run when the cause of action accrues. Under the discovery rule, accrual may be postponed until the plaintiff discovers, or reasonably should discover, the injury.",
    "Equitable tolling may suspend a filing deadline where the defendant fraudulently conceals the existence of a cause of action, or where extraordinary circumstances prevent timely filing and the claimant has acted with reasonable diligence.",
    "Summary judgment is appropriate when there is no genuine dispute as to any material fact and the moving party is entitled to judgment as a matter of law. The court views admissible evidence and reasonable inferences in the nonmovant's favor.",
    "The Due Process Clause generally requires notice and a meaningful opportunity to be heard before the state deprives a person of a protected life, liberty, or property interest.",
    "A valid contract generally requires offer, acceptance, consideration, and sufficiently definite terms. Contract interpretation begins with the language of the agreement.",
    "Res judicata, or claim preclusion, bars a later action between the relevant parties when a final judgment on the merits has already resolved the same claim or a claim that should have been raised.",
    "Collateral estoppel, also called issue preclusion, prevents a party from relitigating an issue that was actually litigated, necessarily decided, and essential to a valid final judgment.",
    "The parol evidence rule may prevent a party from using prior or contemporaneous external statements to contradict the terms of a fully integrated written agreement.",
    "A negligence claim ordinarily requires a duty of care, breach, factual and legal causation, and legally cognizable damages.",
    "An injunction is an equitable remedy that directs a party to do or refrain from doing a specified act. Courts generally consider irreparable harm and the adequacy of legal remedies.",
    "A motion to dismiss for failure to state a claim tests whether the complaint contains sufficient factual allegations to make the requested relief legally plausible.",
    "Prejudgment interest may be available on certain liquidated monetary claims and is often calculated from the date payment became due or the date of breach.",
    "The hearsay rule generally excludes an out-of-court statement offered to prove the truth of the matter asserted, subject to recognized exclusions and exceptions.",
    "A patent claim defines the scope of the claimed invention. Claim construction is a question of law for the court, informed by the claim language and intrinsic patent record.",
]
RELEVANT = [0, 1, 2, 3]
LONG_DOCUMENT = (
    "The interpretation of a statutory limitation period requires analysis of the text, the applicable accrual rule, "
    "and any statutory or equitable tolling provision. A court may need to determine when the claimant knew "
    "or reasonably should have known of the injury, whether the defendant concealed material facts, whether "
    "the claimant exercised diligence, and whether the requested remedy is consistent with the governing "
    "procedural rules. "
) * 12
TEST_DOCUMENTS = DOCUMENTS + [LONG_DOCUMENT]


def log(message: str = "") -> None:
    print(message, flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine().lower(),
        "vendor": "unknown",
        "model": platform.processor() or "unknown",
        "avx2": False,
        "avx512": False,
        "vnni": False,
        "source": "platform",
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            lines = cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines()
            flags_by_cpu: list[set[str]] = []
            for line in lines:
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                key = key.strip().lower()
                value = value.strip()
                if key == "vendor_id" and info["vendor"] == "unknown":
                    info["vendor"] = value
                elif key == "model name" and info["model"].strip().lower() in {"", "unknown", "x86_64", "amd64"}:
                    info["model"] = value
                elif key in {"flags", "features"}:
                    flags_by_cpu.append(set(value.lower().split()))
            if flags_by_cpu:
                # Use the intersection, not the union: features must be
                # available on every logical CPU for safe process-wide use.
                flags = set.intersection(*flags_by_cpu)
                info.update(
                    source="/proc/cpuinfo",
                    avx2="avx2" in flags,
                    avx512=any(flag.startswith("avx512") for flag in flags),
                    vnni=any("vnni" in flag for flag in flags),
                )
        except OSError:
            pass
    return info


def check_environment(summary: dict[str, Any]):
    required = ("torch", "sentence-transformers", "openvino", "openvino-tokenizers", "optimum-intel", "optimum", "transformers", "huggingface-hub")
    missing = [name for name in required if package_version(name) == "missing"]
    if missing:
        raise RuntimeError(
            "Missing packages: " + ", ".join(missing) + ". Install the supplied requirements.txt first."
        )

    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "x64"}:
        raise RuntimeError(f"This Ryzen-focused build expects x86-64; detected platform.machine()={machine!r}")

    import torch
    import openvino as ov

    # TorchVision is not used by this text-only embedding workflow. Some
    # platform/wheel combinations ship an unusable optional TorchVision ops
    # extension; detect that before Transformers triggers a deep import chain.
    try:
        import torchvision  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "torchvision" or (exc.name and exc.name.startswith("torchvision.")):
            log("TorchVision is not installed; continuing (it is not needed for text embeddings).")
        else:
            raise
    except Exception as exc:
        raise RuntimeError(
            "TorchVision is installed but cannot import. This text-only workflow does not need it; "
            "run `python -m pip uninstall -y torchvision` in this virtual environment, then retry. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    import sentence_transformers
    import transformers

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows this setting before parallel work has started.
        pass

    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError(f"OpenVINO CPU plugin is unavailable; detected devices: {core.available_devices}")

    cpu = cpu_info()
    if cpu["vendor"] == "AuthenticAMD":
        log("AMD CPU detected. OpenVINO CPU compatibility will be established by actual IR compile/inference tests.")

    summary["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "openvino": ov.__version__,
        "optimum_intel": package_version("optimum-intel"),
        "optimum": package_version("optimum"),
        "huggingface_hub": package_version("huggingface-hub"),
        "cpu": cpu,
        "openvino_devices": list(core.available_devices),
        "cpu_threads_torch": CPU_THREADS,
        "cpu_threads_openvino": CPU_THREADS,
        "openvino_performance_hint": OV_PERFORMANCE_HINT,
    }
    log(
        f"Python {summary['environment']['python']} | Torch {summary['environment']['torch']} | "
        f"Sentence-Transformers {summary['environment']['sentence_transformers']} | OpenVINO {ov.__version__}"
    )
    log(
        f"CPU: {cpu['vendor']} {cpu['model']} | AVX2={cpu['avx2']} | "
        f"AVX512={cpu['avx512']} | VNNI={cpu['vnni']} | OpenVINO devices={core.available_devices}"
    )
    return torch, ov, core, cpu


def resolve_model_revision() -> str:
    """Resolve a mutable Hub branch/tag to an immutable commit for parity."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
    if not getattr(info, "sha", None):
        raise RuntimeError("Hugging Face did not return a commit SHA for the requested model revision")
    return str(info.sha)


def load_sentence_transformer(
    source: str | Path,
    backend: str,
    torch_module,
    revision: str | None = None,
    export: bool = False,
):
    """Load one Sentence-Transformers model with explicit, verified precision.

    The PyTorch reference is always FP32 compute. The OpenVINO graph is exported
    separately and its stored constants are compressed to FP16 after export.
    """
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"device": "cpu", "backend": backend}
    if revision is not None:
        kwargs["revision"] = revision

    if backend == "torch":
        # The Hub checkpoint may declare BF16. Override that declaration at
        # load time, then .float() as a defensive second step. This ensures
        # FP32 arithmetic rather than BF16 activations rounded to FP32 later.
        kwargs["model_kwargs"] = {"dtype": torch_module.float32}
    elif backend == "openvino":
        # Do not pass a PyTorch dtype here. Export the same checkpoint into
        # OpenVINO, then save its IR with FP16-compressed weights.
        kwargs["model_kwargs"] = {
            "export": bool(export),
            "dynamic_shapes": True,
            "ov_config": {
                "INFERENCE_NUM_THREADS": str(CPU_THREADS),
                "PERFORMANCE_HINT": OV_PERFORMANCE_HINT,
            },
        }
    else:
        raise ValueError(f"Unsupported Sentence-Transformers backend: {backend!r}")

    model = SentenceTransformer(str(source), **kwargs)
    if backend == "torch":
        model.float()
        floating_dtypes = {
            str(tensor.dtype)
            for tensor in list(model.parameters()) + list(model.buffers())
            if getattr(tensor, "is_floating_point", lambda: False)()
        }
        if floating_dtypes != {"torch.float32"}:
            raise RuntimeError(
                "PyTorch reference is not entirely FP32 after explicit conversion; "
                f"floating tensor dtypes={sorted(floating_dtypes)}"
            )
        log("PASS PyTorch reference precision: all floating parameters/buffers are torch.float32")

    model.max_seq_length = MAX_SEQ_LENGTH
    model.eval()
    return model


def declared_checkpoint_dtype(model) -> str | None:
    """Return the backbone config's declared dtype, if it is exposed."""
    for module in model.modules():
        backbone = getattr(module, "auto_model", None)
        config = getattr(backbone, "config", None)
        if config is not None:
            value = getattr(config, "dtype", None)
            if value is None:
                value = getattr(config, "torch_dtype", None)
            if value is not None:
                return str(value)
    return None


def tokenizer_for(model):
    for module in model.modules():
        tokenizer = getattr(module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    raise RuntimeError("Sentence-Transformers model does not expose a tokenizer")


def prompt_signature(model) -> dict[str, Any]:
    prompts = getattr(model, "prompts", {}) or {}
    return {
        "default_prompt_name": getattr(model, "default_prompt_name", None),
        "prompts": {str(key): str(value) for key, value in sorted(prompts.items())},
    }


def tokenizer_check(model, label: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model)
    texts = QUERIES + DOCUMENTS + [LONG_DOCUMENT]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise RuntimeError(f"{label}: tokenizer must return input_ids and attention_mask")
    ids = np.asarray(batch["input_ids"])
    mask = np.asarray(batch["attention_mask"])
    if ids.ndim != 2 or ids.shape[0] != len(texts) or mask.shape != ids.shape:
        raise RuntimeError(f"{label}: invalid tokenizer shapes: input_ids={ids.shape}, attention_mask={mask.shape}")
    lengths = mask.sum(axis=1)
    if np.any(lengths <= 0) or np.any(lengths > MAX_SEQ_LENGTH):
        raise RuntimeError(f"{label}: invalid token lengths: {lengths.tolist()}")
    if not np.isfinite(ids).all() or not np.isin(mask, (0, 1)).all():
        raise RuntimeError(f"{label}: tokenizer emitted non-finite IDs or a non-binary attention mask")
    vocab_size = len(tokenizer)
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise RuntimeError(f"{label}: token ID outside vocabulary size {vocab_size}")

    canonical_ids = ids.astype(np.int64, copy=False)
    canonical_mask = mask.astype(np.int64, copy=False)
    digest = hashlib.sha256(canonical_ids.tobytes() + canonical_mask.tobytes()).hexdigest()
    result = {
        "sha256": digest,
        "texts": len(texts),
        "min_tokens": int(lengths.min()),
        "max_tokens": int(lengths.max()),
        "vocab_size": int(vocab_size),
    }
    log(
        f"PASS {label} tokenizer: {len(texts)} texts; token lengths="
        f"{result['min_tokens']}..{result['max_tokens']}; sha256={digest[:12]}"
    )
    return result


def encode(model, method_name: str, texts: list[str], embedding_dim: int) -> np.ndarray:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise RuntimeError(f"Sentence-Transformers model has no callable {method_name}()")
    values = method(
        list(texts),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        precision="float32",
    )
    values = np.asarray(values, dtype=np.float32)
    expected_shape = (len(texts), embedding_dim)
    if values.shape != expected_shape:
        raise RuntimeError(f"{method_name}: output shape {values.shape}; expected {expected_shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{method_name}: model returned NaN or infinite values")

    # Normalize after converting to float32 so PyTorch and OpenVINO use the
    # same normalization arithmetic and dtype-aware behavior.
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise RuntimeError(f"{method_name}: zero, NaN, or infinite embedding norm")
    values = values / norms
    final_norms = np.linalg.norm(values, axis=1)
    max_norm_error = float(np.max(np.abs(final_norms - 1.0)))
    if max_norm_error > 5e-4:
        raise RuntimeError(
            f"{method_name}: normalized embedding norm error {max_norm_error:.7g} exceeds 0.0005; "
            f"norms={final_norms.tolist()}"
        )
    return np.ascontiguousarray(values, dtype=np.float32)


def cosine_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Embedding shapes differ: {reference.shape} vs {candidate.shape}")
    similarities = np.clip(np.sum(reference * candidate, axis=1), -1.0, 1.0)
    absolute_error = np.abs(reference - candidate)
    return {
        "n": int(len(similarities)),
        "min": float(similarities.min()),
        "mean": float(similarities.mean()),
        "p05": float(np.percentile(similarities, 5)),
        "mae": float(absolute_error.mean()),
        "max_abs_error": float(absolute_error.max()),
    }


def check_padding_invariance(model, label: str, embedding_dim: int) -> dict[str, float]:
    results: dict[str, float] = {}
    tests = (
        ("encode_query", [QUERIES[0], QUERIES[1] + " " + (QUERIES[0] + " ") * 8]),
        ("encode_document", [DOCUMENTS[0], LONG_DOCUMENT]),
    )
    for method_name, texts in tests:
        together = encode(model, method_name, texts, embedding_dim)
        alone = encode(model, method_name, [texts[0]], embedding_dim)[0]
        similarity = float(np.dot(together[0], alone))
        results[method_name] = similarity
        if similarity < 0.999:
            raise RuntimeError(f"{label} {method_name} batch/padding invariance failed: cosine={similarity:.7f}")
    log(f"PASS {label} batch/padding invariance: {results}")
    return results


def check_diversity(query: np.ndarray, docs: np.ndarray, label: str) -> None:
    for name, matrix in (("query", query), ("document", docs)):
        if len(matrix) < 2:
            continue
        similarities = matrix @ matrix.T
        off_diagonal = similarities[~np.eye(len(matrix), dtype=bool)]
        if off_diagonal.size and float(off_diagonal.max()) > 0.99999:
            raise RuntimeError(f"{label} {name} embeddings appear collapsed")


def retrieval_stats(pair: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    query, docs = pair
    scores = query @ docs[: len(DOCUMENTS)].T
    ranking = np.argsort(-scores, axis=1, kind="stable")
    ranks = [int(np.where(ranking[i] == RELEVANT[i])[0][0]) + 1 for i in range(len(QUERIES))]
    return {
        "recall1": float(np.mean([rank <= 1 for rank in ranks])),
        "recall5": float(np.mean([rank <= TOP_K for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
    }


def retrieval_parity(
    reference: tuple[np.ndarray, np.ndarray], candidate: tuple[np.ndarray, np.ndarray]
) -> dict[str, Any]:
    def top(pair):
        query, docs = pair
        return np.argsort(-(query @ docs[: len(DOCUMENTS)].T), axis=1, kind="stable")[:, :TOP_K]

    ref_top = top(reference)
    cur_top = top(candidate)
    jaccard = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ref_top, cur_top)]
    ref_stats = retrieval_stats(reference)
    cur_stats = retrieval_stats(candidate)
    ref_scores = reference[0] @ reference[1][: len(DOCUMENTS)].T
    cur_scores = candidate[0] @ candidate[1][: len(DOCUMENTS)].T
    ref_ranks = [
        int(np.where(row == RELEVANT[i])[0][0]) + 1
        for i, row in enumerate(np.argsort(-ref_scores, axis=1, kind="stable"))
    ]
    cur_ranks = [
        int(np.where(row == RELEVANT[i])[0][0]) + 1
        for i, row in enumerate(np.argsort(-cur_scores, axis=1, kind="stable"))
    ]
    return {
        "recall1_equal": ref_stats["recall1"] == cur_stats["recall1"],
        "recall5_equal": ref_stats["recall5"] == cur_stats["recall5"],
        "relevant_top1_equal": [rank <= 1 for rank in ref_ranks] == [rank <= 1 for rank in cur_ranks],
        "relevant_hits_equal": [rank <= TOP_K for rank in ref_ranks] == [rank <= TOP_K for rank in cur_ranks],
        "jaccard5": float(np.mean(jaccard)),
        "top1": float(np.mean(ref_top[:, 0] == cur_top[:, 0])),
        "mrr_drop": max(0.0, float(ref_stats["mrr"] - cur_stats["mrr"])),
    }


def evaluate_variant(model, label: str, embedding_dim: int):
    tokenizer_info = tokenizer_check(model, label)
    prompts = prompt_signature(model)
    padding = check_padding_invariance(model, label, embedding_dim)

    # Warm-up exercises model compilation and avoids measuring first-run setup.
    encode(model, "encode_query", [QUERIES[0]], embedding_dim)
    encode(model, "encode_document", [DOCUMENTS[0]], embedding_dim)

    timings: list[float] = []
    pair: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        query = encode(model, "encode_query", QUERIES, embedding_dim)
        docs = encode(model, "encode_document", TEST_DOCUMENTS, embedding_dim)
        timings.append(time.perf_counter() - start)
        pair = (query, docs)

    if pair is None:
        raise RuntimeError(f"{label}: evaluation produced no embedding results")
    check_diversity(pair[0], pair[1], label)
    elapsed = float(statistics.median(timings))
    rows = len(QUERIES) + len(TEST_DOCUMENTS)
    info = {
        "tokenizer": tokenizer_info,
        "prompts": prompts,
        "padding": padding,
        "median_seconds": elapsed,
        "rows_per_second": rows / max(elapsed, 1e-9),
        "timings_seconds": timings,
    }
    log(f"PASS {label} encode: {info['rows_per_second']:.2f} rows/s (median of {TIMING_REPEATS} runs)")
    return pair, info



def prepare_fp32_export_checkpoint(torch_model, torch_module) -> tuple[Path, str | None, dict[str, int]]:
    """Serialize the already-loaded FP32 SentenceTransformer as a local FP32 source.

    The Hub revision declares BF16 weights. Optimum's direct export path may preserve
    that dtype, and OpenVINO save_model(compress_to_fp16=True) does not down-convert
    the BF16 constants. Re-serialize the live FP32 model first so the OpenVINO
    exporter starts with FP32 tensors, which its supported FP16 compression handles.
    """
    from safetensors import safe_open

    source_dtype = declared_checkpoint_dtype(torch_model)
    configs_seen = 0
    for module in torch_model.modules():
        backbone = getattr(module, "auto_model", None)
        config = getattr(backbone, "config", None)
        if config is None:
            continue
        try:
            config.dtype = torch_module.float32
        except Exception as exc:
            raise RuntimeError(
                f"Could not set model config dtype to FP32 for {type(backbone).__name__}: {exc}"
            ) from exc
        configs_seen += 1

    if configs_seen == 0:
        raise RuntimeError("Could not find a Transformers backbone config to mark as FP32")

    unique_params = {id(param): param for param in torch_model.parameters()}.values()
    parameter_count = sum(int(param.numel()) for param in unique_params)
    estimated_bytes = parameter_count * 4
    free_bytes = shutil.disk_usage(BASE).free
    required_bytes = int(estimated_bytes * 1.15)
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient disk space to stage the FP32 export source: "
            f"need approximately {required_bytes / (1024**3):.2f} GiB free, "
            f"have {free_bytes / (1024**3):.2f} GiB"
        )

    source_dir = Path(tempfile.mkdtemp(prefix=".dinghy-law-fp32-source-", dir=BASE))
    try:
        log(
            "Stage: serializing the validated FP32 reference as a temporary local export source "
            f"({parameter_count:,} parameters; estimated {estimated_bytes / (1024**3):.2f} GiB)"
        )
        torch_model.save_pretrained(str(source_dir), safe_serialization=True)

        safetensor_files = sorted(source_dir.rglob("*.safetensors"))
        if not safetensor_files:
            raise RuntimeError("FP32 export source contains no safetensors files")

        dtype_counts: dict[str, int] = {}
        for weight_file in safetensor_files:
            with safe_open(str(weight_file), framework="np") as handle:
                for key in handle.keys():
                    dtype_name = str(handle.get_slice(key).get_dtype()).upper()
                    dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        float_dtypes = {key for key in dtype_counts if key in {"F16", "BF16", "F32", "F64"}}
        if not float_dtypes or float_dtypes != {"F32"}:
            raise RuntimeError(
                "Temporary export checkpoint is not exclusively FP32 for floating tensors: "
                f"tensor counts by safetensors dtype={dtype_counts}"
            )

        log(
            "PASS temporary export checkpoint precision: all floating safetensors are F32 "
            f"({sum(dtype_counts.values()):,} tensors; dtype counts={dtype_counts})"
        )
        return source_dir, source_dtype, dtype_counts
    except Exception:
        shutil.rmtree(source_dir, ignore_errors=True)
        raise

def find_openvino_xml(directory: Path) -> Path:
    xmls = sorted(path for path in directory.rglob("*.xml") if path.is_file())
    if len(xmls) != 1:
        relative = [str(path.relative_to(directory)) for path in xmls]
        raise RuntimeError(f"Expected exactly one transformer OpenVINO IR XML; found {len(xmls)}: {relative}")
    return xmls[0]


def inspect_fp16_ir(xml_path: Path, ov) -> dict[str, Any]:
    core = ov.Core()
    model = core.read_model(str(xml_path))
    counts: dict[str, int] = {}
    float_elements: dict[str, int] = {"f16": 0, "f32": 0, "bf16": 0}
    const_nodes = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        const_nodes += 1
        element_type = node.get_output_element_type(0).get_type_name()
        shape = node.get_output_shape(0)
        elements = math.prod(int(dim) for dim in shape)
        counts[element_type] = counts.get(element_type, 0) + elements
        if element_type in float_elements:
            float_elements[element_type] += elements

    float_total = sum(float_elements.values())
    fp16_ratio = float_elements["f16"] / max(float_total, 1)
    result = {
        "xml": str(xml_path),
        "xml_bytes": xml_path.stat().st_size,
        "bin": str(xml_path.with_suffix(".bin")),
        "bin_bytes": xml_path.with_suffix(".bin").stat().st_size if xml_path.with_suffix(".bin").is_file() else 0,
        "constant_nodes": const_nodes,
        "constant_elements_by_type": counts,
        "float_constant_elements": float_elements,
        "fp16_fraction_of_float_constants": fp16_ratio,
        "inputs": [port.get_any_name() for port in model.inputs],
        "outputs": [port.get_any_name() for port in model.outputs],
    }
    del model
    gc.collect()

    if result["bin_bytes"] <= 0:
        raise RuntimeError(f"OpenVINO BIN weights file is missing or empty: {xml_path.with_suffix('.bin')}")
    if float_elements["f16"] <= 0 or fp16_ratio < 0.95:
        raise RuntimeError(
            "OpenVINO IR is not demonstrably FP16-stored: "
            f"FP16 fraction={fp16_ratio:.4f}; constants={float_elements}"
        )
    return result


def compress_ir_weights_to_fp16(xml_path: Path, ov) -> dict[str, Any]:
    """Rewrite IR weights with OpenVINO's supported FP16 weight compression."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    temp_xml = xml_path.with_name(xml_path.stem + ".fp16_staging.xml")
    temp_bin = temp_xml.with_suffix(".bin")
    try:
        for candidate in (temp_xml, temp_bin):
            if candidate.exists():
                candidate.unlink()
        ov.save_model(model, str(temp_xml), compress_to_fp16=True)
        if not temp_xml.is_file() or not temp_bin.is_file() or temp_bin.stat().st_size == 0:
            raise RuntimeError("OpenVINO save_model did not create a complete FP16 XML/BIN pair")
        del model
        gc.collect()

        # Re-open and inspect the staged files before replacing the source pair.
        staged_info = inspect_fp16_ir(temp_xml, ov)
        original_bin = xml_path.with_suffix(".bin")
        os.replace(temp_bin, original_bin)
        os.replace(temp_xml, xml_path)
        log(
            "PASS FP16 IR compression: "
            f"FP16 constant fraction={staged_info['fp16_fraction_of_float_constants']:.4%}; "
            f"weights={original_bin.stat().st_size / (1024 ** 2):.1f} MiB"
        )
        return staged_info
    finally:
        for candidate in (temp_xml, temp_bin):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()


def _input_semantic(port) -> str:
    names = {str(name).lower() for name in port.get_names()}
    any_name = str(port.get_any_name()).lower()
    names.add(any_name)
    for semantic in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        if any(name == semantic or name.endswith("/" + semantic) or name.endswith("." + semantic) for name in names):
            return semantic
    raise RuntimeError(f"Unknown OpenVINO graph input names: {sorted(names)}; refusing to guess input semantics")


def _input_numpy_dtype(port):
    name = port.get_element_type().get_type_name()
    mapping = {"i64": np.int64, "i32": np.int32, "u64": np.uint64, "u32": np.uint32}
    if name not in mapping:
        raise RuntimeError(f"Unexpected input type {name!r} for {port.get_any_name()!r}")
    return mapping[name]


def openvino_graph_smoke_test(xml_path: Path, tokenizer, ov) -> dict[str, Any]:
    """Compile the saved IR on CPU and exercise dynamic batch/sequence shapes."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(
        model,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": CPU_THREADS,
            "PERFORMANCE_HINT": OV_PERFORMANCE_HINT,
        },
    )
    input_ports = list(compiled.inputs)
    if not input_ports:
        raise RuntimeError("OpenVINO IR has no inputs")

    cases = ([QUERIES[0]], [QUERIES[0], LONG_DOCUMENT], [LONG_DOCUMENT])
    observed_shapes: list[dict[str, list[int]]] = []
    for texts in cases:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )
        ids = np.asarray(batch["input_ids"])
        mask = np.asarray(batch["attention_mask"])
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise RuntimeError(f"Tokenizer produced invalid shape for OpenVINO smoke test: {ids.shape}/{mask.shape}")
        feed = {}
        for port in input_ports:
            semantic = _input_semantic(port)
            dtype = _input_numpy_dtype(port)
            if semantic == "input_ids":
                values = ids
            elif semantic == "attention_mask":
                values = mask
            elif semantic == "position_ids":
                # Match the standard mask-aware position numbering used by
                # decoder-only Transformers, including left-padded batches.
                values = np.cumsum(mask.astype(np.int64), axis=1) - 1
                values = np.where(mask.astype(bool), values, 0)
            elif semantic == "token_type_ids":
                values = np.zeros(ids.shape, dtype=np.int64)
            else:
                raise RuntimeError(f"Unhandled OpenVINO input semantic: {semantic}")
            feed[port] = np.asarray(values, dtype=dtype)

        outputs = compiled(feed)
        arrays = [np.asarray(value) for value in outputs.values()]
        token_outputs = [value for value in arrays if value.ndim == 3]
        if not token_outputs:
            raise RuntimeError(f"OpenVINO transformer graph returned no rank-3 token output: {[x.shape for x in arrays]}")
        hidden = token_outputs[0]
        expected = (ids.shape[0], ids.shape[1], EXPECTED_EMBEDDING_DIM)
        if hidden.shape != expected:
            raise RuntimeError(f"OpenVINO token output shape {hidden.shape}; expected {expected}")
        if not np.isfinite(hidden).all():
            raise RuntimeError("OpenVINO CPU graph returned NaN or infinite token embeddings")
        observed_shapes.append({"input_ids": list(ids.shape), "token_output": list(hidden.shape)})

    del compiled, model
    gc.collect()
    if len({tuple(item["input_ids"]) for item in observed_shapes}) < 2:
        raise RuntimeError("OpenVINO smoke test did not exercise multiple batch/sequence shapes")
    log(f"PASS OpenVINO CPU IR compile + dynamic-shape inference: {observed_shapes}")
    return {"device": "CPU", "shapes": observed_shapes}


def verify_artifact_layout(directory: Path) -> tuple[Path, dict[str, Any]]:
    if not (directory / "modules.json").is_file():
        raise RuntimeError(f"Sentence-Transformers modules.json missing from {directory}")
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model"}
    if not any(path.is_file() and path.name in tokenizer_files for path in directory.rglob("*")):
        raise RuntimeError("Exported model is missing tokenizer assets")
    onnx_files = [path for path in directory.rglob("*") if path.is_file() and (path.suffix.lower() == ".onnx" or path.name.endswith(".onnx.data"))]
    if onnx_files:
        raise RuntimeError(f"ONNX files are forbidden in this deliverable: {[str(p) for p in onnx_files]}")
    xml = find_openvino_xml(directory)
    binary = xml.with_suffix(".bin")
    if not binary.is_file() or binary.stat().st_size == 0:
        raise RuntimeError(f"OpenVINO XML/BIN pair incomplete: {xml} / {binary}")
    return xml, {
        "xml_relative": str(xml.relative_to(directory)),
        "bin_relative": str(binary.relative_to(directory)),
        "xml_sha256": sha256_file(xml),
        "bin_sha256": sha256_file(binary),
        "xml_bytes": xml.stat().st_size,
        "bin_bytes": binary.stat().st_size,
    }


def write_manifest(directory: Path, revision: str, artifact_files: dict[str, Any], ir_info: dict[str, Any]) -> None:
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "format": "OpenVINO IR",
        "weight_precision": "FP16",
        "ir_scope": (
            "Transformer encoder IR constants only; Sentence-Transformers pooling, Dense projection, "
            "and normalization remain in the saved Sentence-Transformers pipeline."
        ),
        "quantization": "none",
        "max_seq_length": MAX_SEQ_LENGTH,
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        "artifact_files": artifact_files,
        "ir_validation": ir_info,
    }
    atomic_write_json(directory / MANIFEST_NAME, manifest)


def atomic_install_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".backup")
    if backup.exists() and not destination.exists():
        os.replace(backup, destination)
    elif backup.exists():
        shutil.rmtree(backup)

    moved_old = False
    if destination.exists():
        os.replace(destination, backup)
        moved_old = True
    try:
        os.replace(staging, destination)
    except Exception:
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def cgroup_memory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, path in (
        ("current_bytes", Path("/sys/fs/cgroup/memory.current")),
        ("limit", Path("/sys/fs/cgroup/memory.max")),
        ("events", Path("/sys/fs/cgroup/memory.events")),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if key == "events":
                result[key] = raw
            elif raw.isdigit():
                result[key] = int(raw)
            else:
                result[key] = raw
        except OSError:
            pass
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal", "SwapFree", "SwapTotal"}:
                meminfo[key] = int(value.strip().split()[0]) * 1024
        result["host"] = meminfo
    except (OSError, ValueError):
        pass
    return result


def compare_embedding(summary: dict[str, Any], key: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    stats = cosine_stats(reference, candidate)
    summary.setdefault("cosine", {})[key] = stats
    passed = stats["min"] >= MIN_COSINE and stats["mean"] >= MIN_MEAN_COSINE
    summary.setdefault("checks", []).append({"name": key, "passed": passed, "metrics": stats})
    log(
        f"{'PASS' if passed else 'FAIL'} {key}: min_cosine={stats['min']:.7f}, "
        f"mean_cosine={stats['mean']:.7f}, max_abs_error={stats['max_abs_error']:.6g}"
    )
    if not passed:
        raise RuntimeError(
            f"{key} FP16 embedding fidelity failed: min={stats['min']:.7f} "
            f"(need {MIN_COSINE}), mean={stats['mean']:.7f} (need {MIN_MEAN_COSINE})"
        )


def compare_retrieval(summary: dict[str, Any], key: str, reference, candidate) -> None:
    result = retrieval_parity(reference, candidate)
    summary.setdefault("retrieval", {})[key] = result
    passed = (
        result["recall1_equal"]
        and result["recall5_equal"]
        and result["relevant_top1_equal"]
        and result["relevant_hits_equal"]
        and result["top1"] == 1.0
        and result["jaccard5"] >= MIN_TOP5_JACCARD
        and result["mrr_drop"] <= MAX_MRR_DROP
    )
    summary.setdefault("checks", []).append({"name": key + " retrieval parity", "passed": passed, "metrics": result})
    log(f"{'PASS' if passed else 'FAIL'} {key} retrieval parity: {result}")
    if not passed:
        raise RuntimeError(f"{key} retrieval parity failed")


def main() -> int:
    started = time.perf_counter()
    BASE.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "status": "RUNNING",
        "model": MODEL_ID,
        "format": "OpenVINO IR FP16",
        "quantization": "none",
        "fixture": "synthetic smoke test: 4 queries / 14 passages; not a benchmark or legal-quality evaluation",
        "checks": [],
        "cosine": {},
        "retrieval": {},
        "files_mb": {},
    }
    stage = "environment"
    staging: Path | None = None
    fp32_export_source: Path | None = None
    try:
        summary["script_sha256"] = sha256_file(script_path)
        log(f"SCRIPT: {script_path} | sha256={summary['script_sha256']}")
        torch, ov, core, cpu = check_environment(summary)
        summary["memory_before"] = cgroup_memory()

        stage = "resolve immutable model revision"
        log("Stage: resolving immutable Hugging Face model revision")
        revision = resolve_model_revision()
        summary["model_revision"] = revision
        log(f"Model revision: {revision}")

        stage = "PyTorch FP32 reference"
        log("Stage: loading and validating the PyTorch FP32 reference")
        torch_model = load_sentence_transformer(MODEL_ID, "torch", torch, revision=revision)
        embedding_dim = int(torch_model.get_embedding_dimension() or 0)
        if embedding_dim != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected Sentence-Transformers dimension {embedding_dim}; expected {EXPECTED_EMBEDDING_DIM}"
            )
        base_pair, base_info = evaluate_variant(torch_model, "pytorch_fp32", embedding_dim)
        summary["retrieval"]["pytorch_fp32"] = retrieval_stats(base_pair)
        summary["speed"] = {"pytorch_fp32_rows_per_second": base_info["rows_per_second"]}
        source_declared_dtype = declared_checkpoint_dtype(torch_model)
        summary["reference"] = {
            "execution_dtype": "float32",
            "declared_checkpoint_dtype": source_declared_dtype,
            "checkpoint_weights_note": (
                "The immutable Hub revision declares BF16 weights. The reference loads those values "
                "into FP32 tensors; this does not recreate precision absent from the source checkpoint. "
                "A temporary local FP32 serialization is used as the OpenVINO export source so that "
                "OpenVINO's supported FP32-to-FP16 compression can produce an FP16 IR."
            ),
            "embedding_dimension": embedding_dim,
            "max_seq_length": int(torch_model.max_seq_length),
            "tokenizer": base_info["tokenizer"],
            "prompts": base_info["prompts"],
        }
        fp32_export_source, saved_source_dtype, saved_dtype_counts = prepare_fp32_export_checkpoint(torch_model, torch)
        summary["reference"]["export_source_original_config_dtype"] = saved_source_dtype
        summary["reference"]["export_source_safetensors_dtype_counts"] = saved_dtype_counts
        del torch_model
        gc.collect()
        atomic_write_json(SUMMARY_FILE, summary)

        stage = "OpenVINO FP16 export"
        log("Stage: exporting the validated local FP32 checkpoint to OpenVINO (first export can take several minutes)")
        staging = Path(tempfile.mkdtemp(prefix=".dinghy-openvino-fp16-", dir=BASE))
        log(f"Exporting local FP32 checkpoint from {fp32_export_source} to OpenVINO on CPU ...")
        export_model = load_sentence_transformer(fp32_export_source, "openvino", torch, revision=None, export=True)

        stage = "OpenVINO FP16 weight conversion"
        log("Stage: converting OpenVINO transformer weights to FP16 with Optimum Intel")
        ov_transformer = getattr(export_model, "transformers_model", None)
        half_method = getattr(ov_transformer, "half", None)
        if not callable(half_method):
            raise RuntimeError(
                "SentenceTransformer OpenVINO backend did not expose the Optimum Intel "
                "transformers_model.half() API; refusing to save a possibly BF16 IR"
            )
        half_method()
        log("PASS Optimum Intel transformer_model.half(): requested FP16 weight conversion")

        export_model.save_pretrained(str(staging))
        del ov_transformer, export_model
        gc.collect()

        stage = "OpenVINO IR structure validation"
        xml_path, artifact_files = verify_artifact_layout(staging)
        ir_info = inspect_fp16_ir(xml_path, ov)
        summary["openvino_ir"] = ir_info
        summary["files_mb"]["xml"] = round(artifact_files["xml_bytes"] / (1024 * 1024), 2)
        summary["files_mb"]["bin"] = round(artifact_files["bin_bytes"] / (1024 * 1024), 2)
        if list(staging.rglob("*.onnx")):
            raise RuntimeError("Unexpected ONNX artifact found after OpenVINO export")

        stage = "OpenVINO CPU graph smoke test"
        ov_model = load_sentence_transformer(staging, "openvino", torch, export=False)
        ov_tokenizer = tokenizer_for(ov_model)
        summary["openvino_ir"]["cpu_graph_smoke"] = openvino_graph_smoke_test(xml_path, ov_tokenizer, ov)

        stage = "OpenVINO FP16 Sentence-Transformers validation"
        log("Stage: validating OpenVINO FP16 embeddings and retrieval parity")
        ov_pair, ov_info = evaluate_variant(ov_model, "openvino_fp16_cpu", embedding_dim)
        if ov_info["prompts"] != base_info["prompts"]:
            raise RuntimeError("OpenVINO prompt metadata differs from the PyTorch reference")
        if ov_info["tokenizer"]["sha256"] != base_info["tokenizer"]["sha256"]:
            raise RuntimeError("OpenVINO tokenizer output differs from the PyTorch reference")

        summary["speed"]["openvino_fp16_rows_per_second"] = ov_info["rows_per_second"]
        summary["speed"]["openvino_fp16_vs_pytorch_fp32"] = (
            ov_info["rows_per_second"] / max(base_info["rows_per_second"], 1e-9)
        )
        summary["retrieval"]["openvino_fp16"] = retrieval_stats(ov_pair)
        summary["openvino_validation"] = {
            "tokenizer": ov_info["tokenizer"],
            "prompts": ov_info["prompts"],
            "padding": ov_info["padding"],
            "timings_seconds": ov_info["timings_seconds"],
        }
        compare_embedding(summary, "torch_fp32_vs_openvino_fp16_query", base_pair[0], ov_pair[0])
        compare_embedding(summary, "torch_fp32_vs_openvino_fp16_document", base_pair[1], ov_pair[1])
        compare_retrieval(summary, "torch_fp32_vs_openvino_fp16", base_pair, ov_pair)
        summary["checks"].append({"name": "tokenizer and prompt parity", "passed": True})

        stage = "write manifest and install artifact"
        write_manifest(staging, revision, artifact_files, ir_info)
        atomic_install_directory(staging, OPENVINO_DIR)
        staging = None

        summary["artifact"] = {
            "directory": str(OPENVINO_DIR),
            "xml": artifact_files["xml_relative"],
            "bin": artifact_files["bin_relative"],
            "xml_sha256": artifact_files["xml_sha256"],
            "bin_sha256": artifact_files["bin_sha256"],
            "format": "OpenVINO IR",
            "weight_precision": "FP16",
            "ir_scope": (
                "Transformer encoder IR constants only; pooling, Dense projection, "
                "and normalization remain in the Sentence-Transformers pipeline."
            ),
            "quantization": "none",
        }
        summary["status"] = "PASS"
        log("\nALL CHECKS PASSED: OpenVINO FP16 IR exported and validated on this CPU.")
        log(f"OpenVINO model: {OPENVINO_DIR}")

        del ov_model
        gc.collect()
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["failed_stage"] = stage
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        log(f"\nFAILED during {stage}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if fp32_export_source is not None and fp32_export_source.exists():
            shutil.rmtree(fp32_export_source, ignore_errors=True)
        summary["memory_after"] = cgroup_memory()
        summary["seconds"] = round(time.perf_counter() - started, 2)
        try:
            atomic_write_json(SUMMARY_FILE, summary)
            log(f"Validation report: {SUMMARY_FILE}")
        except Exception as exc:
            log(f"Could not write validation report: {type(exc).__name__}: {exc}")

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("This script takes no command-line arguments. Run: python convert_and_test.py", flush=True)
        raise SystemExit(2)
    raise SystemExit(main())
