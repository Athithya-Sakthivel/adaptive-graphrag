#!/usr/bin/env python3
"""Export and validate dinghy-law-0.6b-v1 with FP32 ONNX and weight-only INT8 MatMulNBits.
Run: python3 convert_and_test.py (no arguments). Writes a human-readable JSON summary.
"""
from __future__ import annotations

import gc, hashlib, importlib.metadata, inspect, json, os, platform, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np

MODEL_ID = "Hanno-Labs/dinghy-law-0.6b-v1"
MODEL_REVISION = os.getenv("MODEL_REVISION") or None
MAX_SEQ_LENGTH = 8192
EMBEDDING_DIM = 1024
BATCH_SIZE = 2
WEIGHT_BITS = 8
WEIGHT_BLOCK_SIZE = 32
WEIGHT_ACCURACY_LEVEL = 1
CPU_PROVIDER = "CPUExecutionProvider"
CPU_THREADS = max(1, int(os.getenv("OMP_NUM_THREADS", min(8, os.cpu_count() or 4))))
MIN_COSINE = 0.990
MIN_MEAN_COSINE = 0.995
MIN_TOP5_JACCARD = 0.90
MAX_MRR_DROP = 0.05
TOP_K = 5
INT8_SUFFIX = "weightonly_int8_b32_a1"
MANIFEST_NAME = "weightonly_int8_manifest.json"
BASE = Path(__file__).resolve().parent
FP32_DIR = BASE / "dinghy_law_onnx"
INT8_DIR = BASE / "dinghy_law_onnx_int8_weight_only"
SUMMARY_FILE = BASE / "model_comparison_summary.json"
TOKENIZER_REF = None

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
LONG_DOCUMENT = ("The interpretation of a statutory limitation period requires analysis of the text, the applicable accrual rule, "
                 "and any statutory or equitable tolling provision. A court may need to determine when the claimant knew "
                 "or reasonably should have known of the injury, whether the defendant concealed material facts, whether "
                 "the claimant exercised diligence, and whether the requested remedy is consistent with the governing "
                 "procedural rules. ") * 35
TEST_DOCUMENTS = DOCUMENTS + [LONG_DOCUMENT]


def log(message=""):
    print(message, flush=True)


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return options


def cpu_features():
    machine = platform.machine().lower()
    found = {"machine": machine, "vendor": "unknown", "model": "unknown",
             "avx2": False, "avx512": False, "vnni": False, "source": "unknown"}
    path = Path("/proc/cpuinfo")
    if path.is_file():
        try:
            text = path.read_text(errors="ignore")
            flag_sets = []
            for line in text.splitlines():
                low = line.lower()
                if low.startswith("vendor_id") and ":" in line and found["vendor"] == "unknown":
                    found["vendor"] = line.split(":", 1)[1].strip()
                elif low.startswith("model name") and ":" in line and found["model"] == "unknown":
                    found["model"] = line.split(":", 1)[1].strip()
                elif low.startswith(("flags", "features")) and ":" in line:
                    flag_sets.append(set(line.split(":", 1)[1].lower().split()))
            flags = set.intersection(*flag_sets) if flag_sets else set()
            if flags:
                found.update(source="/proc/cpuinfo", avx2="avx2" in flags,
                             avx512=any(x.startswith("avx512") for x in flags),
                             vnni=any("vnni" in x for x in flags))
                return found
        except OSError:
            pass
    try:
        from numpy._core._multiarray_umath import __cpu_features__
        features = {str(k).upper(): bool(v) for k, v in __cpu_features__.items()}
        found.update(source="NumPy", avx2=features.get("AVX2", False),
                     avx512=any(v for k, v in features.items() if k.startswith("AVX512")),
                     vnni=any(v for k, v in features.items() if "VNNI" in k))
    except Exception:
        pass
    return found


def check_environment(summary):
    required = ("torch", "onnxruntime", "transformers", "sentence-transformers",
                "optimum-onnx", "onnx", "onnx-ir", "ml-dtypes")
    missing = [name for name in required if version(name) == "missing"]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing) +
                           '. Install with: python -m pip install -U "sentence-transformers[onnx]" onnx onnx-ir ml-dtypes')
    if version("onnxruntime") != "1.31.1":
        raise RuntimeError(f"This script targets the verified ONNX Runtime 1.31.1 MatMulNBits API; found {version('onnxruntime')}. Install with: python -m pip install onnxruntime==1.31.1")
    import torch, onnxruntime as ort, transformers, sentence_transformers
    import onnx_ir, ml_dtypes
    if CPU_PROVIDER not in ort.get_available_providers():
        raise RuntimeError(f"{CPU_PROVIDER} unavailable: {ort.get_available_providers()}")
    torch.set_num_threads(CPU_THREADS)
    cpu = cpu_features()
    summary["env"] = {"python": sys.version.split()[0], "torch": version("torch"),
                      "ort": version("onnxruntime"), "st": sentence_transformers.__version__,
                      "optimum": version("optimum-onnx"), "onnx_ir": version("onnx-ir"),
                      "ml_dtypes": version("ml-dtypes"), "cpu": cpu["machine"],
                      "vendor": cpu["vendor"], "model": cpu["model"],
                      "avx2": cpu["avx2"], "avx512": cpu["avx512"], "vnni": cpu["vnni"],
                      "threads": CPU_THREADS}
    log(f"Python {summary['env']['python']} | Torch {summary['env']['torch']} | ORT {summary['env']['ort']} | ST {summary['env']['st']}")
    log(f"CPU: {cpu['vendor']} {cpu['model']} | AVX2={cpu['avx2']} AVX512={cpu['avx512']} VNNI={cpu['vnni']}")
    return torch, ort, cpu


def load_model(path, backend, ort=None, file_name=None, export=None):
    from sentence_transformers import SentenceTransformer
    kwargs = {"device": "cpu", "backend": backend,
              "processor_kwargs": {"fix_mistral_regex": True}}
    if MODEL_REVISION and not Path(str(path)).exists():
        kwargs["revision"] = MODEL_REVISION
    if backend == "onnx":
        model_kwargs = {"provider": CPU_PROVIDER, "session_options": session_options(ort)}
        if file_name:
            model_kwargs["file_name"] = file_name
        if export is not None:
            model_kwargs["export"] = export
        kwargs["model_kwargs"] = model_kwargs
    model = SentenceTransformer(str(path), **kwargs)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def find_fp32_graph(directory):
    graphs = sorted(p for p in directory.rglob("*.onnx") if p.is_file()
                    and not any(x in p.name.lower() for x in ("qint8", "int8", "quantized")))
    preferred = [p for p in graphs if p.relative_to(directory).as_posix() in {"onnx/model.onnx", "model.onnx"}]
    if len(preferred) == 1:
        return preferred[0]
    if len(graphs) != 1:
        names = [p.relative_to(directory).as_posix() for p in graphs]
        raise RuntimeError(f"Expected one unambiguous FP32 ONNX graph under {directory}; found {names}")
    return graphs[0]


def complete_st_model(directory):
    if not directory.is_dir() or not (directory / "modules.json").is_file():
        return False
    has_tokenizer = any(p.name in {"tokenizer.json", "tokenizer_config.json"} for p in directory.rglob("*"))
    if not has_tokenizer:
        return False
    try:
        find_fp32_graph(directory)
        return True
    except RuntimeError:
        return False


def atomic_replace(staging, destination):
    backup = destination.with_name(destination.name + ".backup")
    if backup.exists():
        shutil.rmtree(backup)
    moved = False
    if destination.exists():
        os.replace(destination, backup)
        moved = True
    try:
        os.replace(staging, destination)
    except Exception:
        if moved and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def prepare_fp32(ort):
    if complete_st_model(FP32_DIR):
        graph = find_fp32_graph(FP32_DIR)
        log(f"Reusing FP32 ONNX: {graph}")
        return graph
    BASE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dinghy-fp32-", dir=BASE))
    try:
        log(f"Exporting {MODEL_ID} to ONNX (first run only) ...")
        model = load_model(MODEL_ID, "onnx", ort, export=True)
        model.save_pretrained(str(staging))
        del model
        gc.collect()
        if not complete_st_model(staging):
            raise RuntimeError("FP32 export incomplete: expected modules.json, tokenizer assets and one ONNX graph")
        atomic_replace(staging, FP32_DIR)
        graph = find_fp32_graph(FP32_DIR)
        log(f"FP32 ONNX installed: {graph}")
        return graph
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def tokenizer_for(model):
    for module in model.modules():
        tok = getattr(module, "tokenizer", None)
        if tok is not None:
            return tok
    raise RuntimeError("Sentence Transformers model has no accessible tokenizer")


def prompt_signature(model):
    prompts = getattr(model, "prompts", {}) or {}
    return {"default": getattr(model, "default_prompt_name", None),
            "prompts": {str(k): str(v) for k, v in sorted(prompts.items())}}


def tokenizer_check(model, label):
    tok = tokenizer_for(model)
    texts = QUERIES + DOCUMENTS + [LONG_DOCUMENT]
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="np")
    ids, mask = np.asarray(batch["input_ids"]), np.asarray(batch["attention_mask"])
    if ids.ndim != 2 or ids.shape[0] != len(texts) or mask.shape != ids.shape:
        raise RuntimeError(f"{label}: tokenizer returned invalid shapes {ids.shape}/{mask.shape}")
    lengths = mask.sum(axis=1)
    if (np.any(lengths <= 0) or np.any(lengths > MAX_SEQ_LENGTH)
            or not np.isfinite(ids).all() or not np.isin(mask, (0, 1)).all()):
        raise RuntimeError(f"{label}: invalid token IDs, attention mask or lengths")
    if ids.min() < 0 or ids.max() >= len(tok):
        raise RuntimeError(f"{label}: tokenizer emitted IDs outside vocabulary size {len(tok)}")
    digest = hashlib.sha256(ids.tobytes() + mask.tobytes()).hexdigest()
    result = {"sha256": digest, "max_tokens": int(lengths.max())}
    log(f"PASS {label} tokenizer: {len(texts)} texts; tokens={int(lengths.min())}..{int(lengths.max())}; sha256={digest[:12]}")
    return result


def encode(model, method_name, texts):
    method = getattr(model, method_name, None)
    if not callable(method):
        raise RuntimeError(f"SentenceTransformer has no {method_name}()")
    values = method(list(texts), batch_size=BATCH_SIZE, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True, precision="float32")
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(texts), EMBEDDING_DIM) or not np.isfinite(values).all():
        raise RuntimeError(f"{method_name}: invalid embedding output {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1e-12) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError(f"{method_name}: invalid embedding norms {norms.min()}..{norms.max()}")
    return values


def cosine_stats(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f"Embedding shapes differ: {a.shape} vs {b.shape}")
    cosine = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    diff = np.abs(a - b)
    return {"n": int(len(cosine)), "min": float(cosine.min()), "mean": float(cosine.mean()),
            "p05": float(np.percentile(cosine, 5)), "mae": float(diff.mean())}


def check_padding_invariance(model, label):
    cases = (("encode_query", [QUERIES[0], QUERIES[1] + " " + QUERIES[0] * 8]),
             ("encode_document", [DOCUMENTS[0], LONG_DOCUMENT]))
    values = {}
    for method, texts in cases:
        together = encode(model, method, texts)
        alone = encode(model, method, [texts[0]])[0]
        similarity = float(np.dot(together[0], alone))
        values[method] = similarity
        if similarity < 0.999:
            raise RuntimeError(f"{label} {method} batch/padding invariance failed: {similarity:.7f}")
    log(f"PASS {label} batch/padding invariance: {values}")
    return values


def check_diversity(query, docs, label):
    for name, matrix in (("query", query), ("document", docs)):
        sim = matrix @ matrix.T
        mask = ~np.eye(len(matrix), dtype=bool)
        if mask.any() and float(sim[mask].max()) > 0.99999:
            raise RuntimeError(f"{label} {name} embeddings appear collapsed")


def graph_smoke_test(ort, graph, tokenizer, label):
    session = ort.InferenceSession(str(graph), sess_options=session_options(ort), providers=[CPU_PROVIDER])
    try:
        providers = session.get_providers()
        if not providers or providers[0] != CPU_PROVIDER:
            raise RuntimeError(f"{label}: graph not running on CPU: {providers}")
        inputs, outputs = {x.name: x for x in session.get_inputs()}, session.get_outputs()
        if not outputs or not {"input_ids", "attention_mask"}.issubset(inputs):
            raise RuntimeError(f"{label}: unexpected graph inputs/outputs: {list(inputs)}")
        for name in ("input_ids", "attention_mask"):
            shape = inputs[name].shape
            if len(shape) != 2 or isinstance(shape[0], int) or isinstance(shape[1], int):
                raise RuntimeError(f"{label}: {name} is not dynamic in batch and sequence axes: {shape}")
        out_shape = outputs[0].shape
        if len(out_shape) != 3 or isinstance(out_shape[0], int) or isinstance(out_shape[1], int):
            raise RuntimeError(f"{label}: token output is not dynamic in batch/sequence axes: {out_shape}")
        tests = ([QUERIES[0]], [QUERIES[0], DOCUMENTS[0]], [LONG_DOCUMENT])
        shapes = []
        for texts in tests:
            batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="np")
            ids, mask = np.asarray(batch["input_ids"]), np.asarray(batch["attention_mask"])
            feed = {}
            for name, spec in inputs.items():
                dtype = np.int64 if "int64" in spec.type else np.int32
                if name == "input_ids":
                    feed[name] = ids.astype(dtype, copy=False)
                elif name == "attention_mask":
                    feed[name] = mask.astype(dtype, copy=False)
                elif name == "position_ids":
                    feed[name] = np.broadcast_to(np.arange(ids.shape[1], dtype=dtype)[None, :], ids.shape).copy()
                elif name == "token_type_ids":
                    feed[name] = np.zeros(ids.shape, dtype=dtype)
                else:
                    raise RuntimeError(f"{label}: unknown graph input {name!r}; refusing to guess")
            output = np.asarray(session.run([outputs[0].name], feed)[0])
            expected = (ids.shape[0], ids.shape[1], EMBEDDING_DIM)
            if output.shape != expected or not np.isfinite(output).all():
                raise RuntimeError(f"{label}: graph output {output.shape}, expected {expected}")
            shapes.append(list(ids.shape))
        log(f"PASS {label} graph CPU/dynamic-shape smoke: {shapes}")
        return {"inputs": list(inputs), "shapes": shapes}
    finally:
        del session
        gc.collect()


def position_ids_adapter(model, input_names, torch, label):
    if "position_ids" not in input_names:
        return 0
    count = 0
    for module in model.modules():
        wrapped = getattr(module, "model", None)
        original = getattr(wrapped, "forward", None)
        if wrapped is None or not callable(original):
            continue
        try:
            sig = inspect.signature(original)
        except (TypeError, ValueError):
            continue
        accepts = "position_ids" in sig.parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts:
            continue
        params = getattr(module, "model_forward_params", None)
        if params is not None and "position_ids" not in params:
            module.model_forward_params = set(params) | {"position_ids"}
        if not getattr(wrapped, "_dinghy_position_ids_adapter", False):
            def forward_with_positions(*args, _original=original, _torch=torch, **kwargs):
                ids = kwargs.get("input_ids", args[0] if args else None)
                if kwargs.get("position_ids") is None:
                    if ids is None or not hasattr(ids, "shape") or len(ids.shape) != 2:
                        raise RuntimeError("Cannot infer position_ids without input_ids [batch, sequence]")
                    batch, seq = map(int, ids.shape)
                    if isinstance(ids, _torch.Tensor):
                        pos = _torch.arange(seq, dtype=_torch.long, device=ids.device)[None, :].expand(batch, -1).contiguous()
                    else:
                        pos = np.broadcast_to(np.arange(seq, dtype=np.int64)[None, :], (batch, seq)).copy()
                    kwargs["position_ids"] = pos
                return _original(*args, **kwargs)
            wrapped.forward = forward_with_positions
            wrapped._dinghy_position_ids_adapter = True
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        output = wrapped(input_ids=ids, attention_mask=torch.ones_like(ids))
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, dict):
            hidden = output.get("last_hidden_state")
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if hidden is None or tuple(hidden.shape) != (1, 4, EMBEDDING_DIM):
            raise RuntimeError(f"{label}: position_ids adapter smoke test failed")
        count += 1
    if not count:
        raise RuntimeError(f"{label}: graph requires position_ids but no compatible forward adapter was found")
    log(f"PASS {label} position_ids adapter ({count} module(s))")
    return count


def evaluate_variant(label, directory, backend, ort, torch, graph=None, file_name=None):
    global TOKENIZER_REF
    if graph is not None:
        if TOKENIZER_REF is None:
            raise RuntimeError("Reference tokenizer unavailable before ONNX graph validation")
        graph_info = graph_smoke_test(ort, graph, TOKENIZER_REF, label)
        input_names = set(graph_info["inputs"])
    else:
        graph_info, input_names = None, set()
    model = load_model(directory, backend, ort, file_name=file_name, export=False if backend == "onnx" else None)
    adapter_count = position_ids_adapter(model, input_names, torch, label) if graph is not None else 0
    tok_info = tokenizer_check(model, label)
    if backend == "torch":
        TOKENIZER_REF = tokenizer_for(model)
    prompts = prompt_signature(model)
    padding = check_padding_invariance(model, label)
    start = time.perf_counter()
    query = encode(model, "encode_query", QUERIES)
    docs = encode(model, "encode_document", TEST_DOCUMENTS)
    elapsed = time.perf_counter() - start
    check_diversity(query, docs, label)
    result = {"tokenizer": tok_info, "prompts": prompts, "padding": padding,
              "seconds": elapsed, "rows_per_second": (len(QUERIES) + len(TEST_DOCUMENTS)) / max(elapsed, 1e-9),
              "graph": graph_info, "adapter": adapter_count}
    del model
    gc.collect()
    return (query, docs), result


def retrieval_stats(pair):
    query, docs = pair
    scores = query @ docs[:len(DOCUMENTS)].T
    ranking = np.argsort(-scores, axis=1, kind="stable")
    ranks = [int(np.where(ranking[i] == RELEVANT[i])[0][0]) + 1 for i in range(len(QUERIES))]
    return {"recall1": float(np.mean([r <= 1 for r in ranks])),
            "recall5": float(np.mean([r <= TOP_K for r in ranks])),
            "mrr": float(np.mean([1.0 / r for r in ranks]))}


def retrieval_parity(reference, candidate):
    def top(pair):
        q, d = pair
        return np.argsort(-(q @ d[:len(DOCUMENTS)].T), axis=1, kind="stable")[:, :TOP_K]
    ref_top, cur_top = top(reference), top(candidate)
    jaccard = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ref_top, cur_top)]
    ref, cur = retrieval_stats(reference), retrieval_stats(candidate)
    ref_q, cur_q = reference[0] @ reference[1][:len(DOCUMENTS)].T, candidate[0] @ candidate[1][:len(DOCUMENTS)].T
    ref_ranks = [int(np.where(row == RELEVANT[i])[0][0]) + 1 for i, row in enumerate(np.argsort(-ref_q, axis=1, kind="stable"))]
    cur_ranks = [int(np.where(row == RELEVANT[i])[0][0]) + 1 for i, row in enumerate(np.argsort(-cur_q, axis=1, kind="stable"))]
    return {"recall1_equal": ref["recall1"] == cur["recall1"],
            "recall5_equal": ref["recall5"] == cur["recall5"],
            "relevant_top1_equal": [r <= 1 for r in ref_ranks] == [r <= 1 for r in cur_ranks],
            "relevant_hits_equal": [r <= TOP_K for r in ref_ranks] == [r <= TOP_K for r in cur_ranks],
            "jaccard5": float(np.mean(jaccard)), "top1": float(np.mean(ref_top[:, 0] == cur_top[:, 0])),
            "mrr_drop": max(0.0, float(ref["mrr"] - cur["mrr"]))}


def find_int8_graph(directory):
    paths = sorted(directory.rglob(f"*_{INT8_SUFFIX}.onnx")) if directory.is_dir() else []
    if len(paths) > 1:
        raise RuntimeError(f"Multiple weight-only INT8 graphs found: {paths}")
    return paths[0] if paths else None


def _onnx_external_paths(graph):
    """Resolve every external initializer file without loading its tensor bytes."""
    import onnx
    from onnx import external_data_helper

    graph = Path(graph).resolve()
    model = onnx.load(str(graph), load_external_data=False)
    root = graph.parent.resolve()
    paths = set()
    for tensor in model.graph.initializer:
        if not external_data_helper.uses_external_data(tensor):
            continue
        location = next((entry.value for entry in tensor.external_data if entry.key == "location"), None)
        if not location:
            raise RuntimeError(f"External tensor {tensor.name!r} has no location")
        resolved = (root / location).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError(f"Invalid or missing external tensor data: {location!r}")
        paths.add(resolved)
    del model
    return sorted(paths)


def _onnx_bundle(graph):
    graph = Path(graph).resolve()
    return [graph, *_onnx_external_paths(graph)]


def _artifact_size_bytes(graph):
    return sum(path.stat().st_size for path in _onnx_bundle(graph))


def _artifact_digest(graph):
    digest = hashlib.sha256()
    for path in _onnx_bundle(graph):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _copytree_ignore_onnx_source(source_graph, external_paths):
    excluded = {Path(source_graph).resolve(), *(Path(item).resolve() for item in external_paths)}
    def ignore(directory, names):
        base = Path(directory).resolve()
        return [name for name in names if (base / name).resolve() in excluded]
    return ignore


def _cgroup_memory():
    result = {}
    for key, path in (("current", "/sys/fs/cgroup/memory.current"),
                      ("max", "/sys/fs/cgroup/memory.max"),
                      ("events", "/sys/fs/cgroup/memory.events"),
                      ("v1_limit", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                      ("v1_usage", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            value = Path(path).read_text().strip()
            result[key] = value if key == "events" else (int(value) if value.isdigit() else value)
        except OSError:
            pass
    try:
        data = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal", "SwapFree", "SwapTotal"}:
                data[key] = int(value.strip().split()[0]) * 1024
        result["host"] = data
    except (OSError, ValueError):
        pass
    return result

def _weightonly_quantize_worker(source, destination):
    """Quantize MatMul weights to 8-bit and validate the saved ONNX artifact."""
    import resource
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
        QuantFormat,
    )

    src = Path(source).resolve()
    dst = Path(destination).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    external = _onnx_external_paths(src)
    linked = [p for p in [src, *external] if p.stat().st_nlink > 1]

    log(
        f"WORKER: MatMulNBits INT8 | "
        f"external_files={len(external)} | "
        f"multiply_linked={len(linked)}"
    )

    private_context = (
        tempfile.TemporaryDirectory(prefix="dinghy-private-")
        if linked else None
    )

    try:
        input_graph = src

        if private_context:
            root = Path(private_context.name)
            input_graph = root / src.name
            shutil.copy2(src, input_graph)

            for external_file in external:
                target = root / external_file.relative_to(src.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(external_file, target)

        config = DefaultWeightOnlyQuantConfig(
            block_size=WEIGHT_BLOCK_SIZE,
            is_symmetric=True,
            accuracy_level=WEIGHT_ACCURACY_LEVEL,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            bits=WEIGHT_BITS,
        )

        model = onnx.load(str(input_graph), load_external_data=True)

        quantizer = MatMulNBitsQuantizer(
            model=model,
            bits=WEIGHT_BITS,
            block_size=WEIGHT_BLOCK_SIZE,
            is_symmetric=True,
            accuracy_level=WEIGHT_ACCURACY_LEVEL,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            algo_config=config,
        )

        del model
        quantizer.process()

        # Save graph and external tensors.
        quantizer.model.save_model_to_file(str(dst), True)

        del quantizer
        gc.collect()

    finally:
        if private_context:
            private_context.cleanup()

    # Do NOT impose an arbitrary 1 KB minimum.
    # Small valid ONNX graphs may store all tensors inline.
    if not dst.is_file():
        raise RuntimeError(
            f"Quantizer did not create output graph: {dst}"
        )

    if dst.stat().st_size == 0:
        raise RuntimeError(
            f"Quantizer created an empty ONNX graph: {dst}"
        )

    # Validate protobuf and external tensor data.
    try:
        saved_model = onnx.load(
            str(dst),
            load_external_data=True,
        )
        onnx.checker.check_model(
            saved_model,
            full_check=False,
        )
        del saved_model
        gc.collect()
    except Exception as exc:
        raise RuntimeError(
            f"Saved ONNX model is invalid or has missing "
            f"external tensor data: {exc}"
        ) from exc

    # Validate actual MatMulNBits operators and packed weights.
    report = inspect_weightonly_graph(dst)

    if report["matmul_nbits_nodes"] < 1:
        raise RuntimeError(
            "Saved graph contains no MatMulNBits operators"
        )

    peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    peak_mib = peak / (
        1024 if sys.platform != "darwin"
        else 1024 * 1024
    )

    log(
        f"WORKER PASS: graph={dst.stat().st_size} bytes; "
        f"external data={Path(str(dst) + '.data').exists()}; "
        f"MatMulNBits={report['matmul_nbits_nodes']}; "
        f"peak RSS={peak_mib:.0f} MiB"
    )

def inspect_weightonly_graph(graph):
    """Check MatMulNBits QOperator nodes, packed 8-bit weights, scales and FP32 activations."""
    import onnx
    model = onnx.load(str(graph), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    counts = {}
    nodes = []
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        if node.op_type == "MatMulNBits" and node.domain == "com.microsoft":
            nodes.append(node)
    if not nodes:
        raise RuntimeError(f"No com.microsoft::MatMulNBits operators found: {counts}")

    packed_names, scale_names = set(), set()
    for node in nodes:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        if int(attrs.get("bits", -1)) != 8:
            raise RuntimeError(f"{node.name}: expected 8-bit MatMulNBits, got attributes {attrs}")
        if int(attrs.get("block_size", -1)) != WEIGHT_BLOCK_SIZE:
            raise RuntimeError(f"{node.name}: unexpected block_size in {attrs}")
        if int(attrs.get("accuracy_level", -1)) != WEIGHT_ACCURACY_LEVEL:
            raise RuntimeError(f"{node.name}: expected accuracy_level=1 (FP32 activations), got {attrs}")
        if len(node.input) < 3:
            raise RuntimeError(f"{node.name}: MatMulNBits has too few inputs")
        packed = initializers.get(node.input[1])
        scales = initializers.get(node.input[2])
        if packed is None or packed.data_type != onnx.TensorProto.UINT8:
            raise RuntimeError(f"{node.name}: packed 8-bit weight initializer missing or not UINT8")
        if scales is None or scales.data_type not in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}:
            raise RuntimeError(f"{node.name}: floating-point block scales missing")
        k_dim, n_dim = int(attrs.get("K", 0)), int(attrs.get("N", 0))
        if k_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{node.name}: missing/invalid K or N attributes: {attrs}")
        k_blocks = (k_dim + WEIGHT_BLOCK_SIZE - 1) // WEIGHT_BLOCK_SIZE
        expected_packed = [n_dim, k_blocks, WEIGHT_BLOCK_SIZE]
        if list(packed.dims) != expected_packed:
            raise RuntimeError(f"{node.name}: packed weight shape {list(packed.dims)} != {expected_packed}")
        valid_scale_shapes = ([n_dim, k_blocks], [n_dim * k_blocks])
        if list(scales.dims) not in valid_scale_shapes:
            raise RuntimeError(f"{node.name}: scales shape {list(scales.dims)} not in {valid_scale_shapes}")
        if len(node.input) > 3 and node.input[3]:
            raise RuntimeError(f"{node.name}: symmetric quantization must not have a zero-point input")
        packed_names.add(packed.name)
        scale_names.add(scales.name)

    dynamic_ops = sum(counts.get(op, 0) for op in ("DynamicQuantizeLinear", "MatMulInteger", "MatMulIntegerToFloat"))
    if dynamic_ops:
        raise RuntimeError(f"Weight-only graph unexpectedly contains dynamic activation quantization operators: {dynamic_ops}")
    remaining = counts.get("MatMul", 0)
    return {
        "matmul_nbits_nodes": len(nodes),
        "packed_weight_initializers": len(packed_names),
        "block_scale_initializers": len(scale_names),
        "bits": 8,
        "block_size": WEIGHT_BLOCK_SIZE,
        "accuracy_level": WEIGHT_ACCURACY_LEVEL,
        "activation_compute": "FP32",
        "remaining_fp32_matmul_nodes": remaining,
        "operator_counts": counts,
    }


def _run_weightonly_worker(source, destination):
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                "LOKY_MAX_CPU_COUNT": "1", "JOBLIB_MULTIPROCESSING": "0", "MALLOC_ARENA_MAX": "2"})
    command = [sys.executable, str(Path(__file__).resolve()), "--_weightonly_worker", str(source), str(destination)]
    log("Starting isolated ONNX Runtime weight-only INT8 worker (no calibration).")
    log(f"Memory before worker: {_cgroup_memory()}")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        if completed.returncode in {-9, 137}:
            raise RuntimeError("Weight-only INT8 worker was SIGKILLed (exit 137/-9); inspect host OOM records. FP32 remains untouched.")
        raise RuntimeError(f"Weight-only INT8 worker exited with code {completed.returncode}")


def preflight_matmul_nbits(ort):
    """Test installed ORT MatMulNBits CPU kernel and FP32-activation semantics on a tiny graph."""
    import onnx
    rng = np.random.default_rng(19)
    with tempfile.TemporaryDirectory(prefix="dinghy-nbits-preflight-") as tmp:
        root = Path(tmp)
        x_path, q_path = root / "toy_fp32.onnx", root / "toy_int8.onnx"
        weight = rng.normal(0.0, 0.2, size=(32, 16)).astype(np.float32)
        x_info = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, ["batch", 32])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["batch", 16])
        initializer = onnx.numpy_helper.from_array(weight, name="B")
        graph = onnx.helper.make_graph([onnx.helper.make_node("MatMul", ["A", "B"], ["Y"])],
                                       "weight_only_preflight", [x_info], [y_info], [initializer])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)],
                                       ir_version=9)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save_model(model, x_path)
        del model
        _weightonly_quantize_worker(str(x_path), str(q_path))
        q_report = inspect_weightonly_graph(q_path)
        x = rng.normal(size=(2, 32)).astype(np.float32)
        options = session_options(ort)
        fp_session = ort.InferenceSession(str(x_path), sess_options=options, providers=[CPU_PROVIDER])
        q_session = ort.InferenceSession(str(q_path), sess_options=options, providers=[CPU_PROVIDER])
        try:
            ref = fp_session.run(None, {"A": x})[0]
            batch = q_session.run(None, {"A": x})[0]
            alone = q_session.run(None, {"A": x[:1]})[0]
            cosine = float(np.dot(batch[0], ref[0]) / max(np.linalg.norm(batch[0]) * np.linalg.norm(ref[0]), 1e-12))
            batch_cosine = float(np.dot(batch[0], alone[0]) / max(np.linalg.norm(batch[0]) * np.linalg.norm(alone[0]), 1e-12))
            if not np.isfinite(batch).all() or cosine < 0.99 or batch_cosine < 0.99999:
                raise RuntimeError(f"MatMulNBits CPU preflight failed: cosine={cosine:.7f}, batch_cosine={batch_cosine:.7f}")
        finally:
            del fp_session, q_session
            gc.collect()
        log(f"PASS MatMulNBits CPU preflight: fp32_cosine={cosine:.7f}; batch_cosine={batch_cosine:.7f}")
        return {"fp32_cosine": round(cosine, 7), "batch_cosine": round(batch_cosine, 7),
                "matmul_nbits_nodes": q_report["matmul_nbits_nodes"]}


def prepare_int8(ort, cpu):
    """Create an isolated 8-bit weight-only MatMulNBits model; never modify FP32."""
    if cpu["machine"] not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        raise RuntimeError(f"MatMulNBits CPU preset is intended for x86/x64; got {cpu['machine']}")
    preflight = preflight_matmul_nbits(ort)
    source_graph = find_fp32_graph(FP32_DIR)
    source_digest = _artifact_digest(source_graph)
    expected = {"schema": 7, "model": MODEL_ID, "revision": MODEL_REVISION,
                "source_sha256": source_digest, "suffix": INT8_SUFFIX,
                "scheme": "RTN weight-only MatMulNBits", "bits": 8,
                "block_size": WEIGHT_BLOCK_SIZE, "symmetric": True,
                "accuracy_level": WEIGHT_ACCURACY_LEVEL, "activation_compute": "FP32",
                "quant_format": "QOperator", "op_types": ["MatMul"],
                "ort": version("onnxruntime")}
    cached = find_int8_graph(INT8_DIR)
    manifest_path = INT8_DIR / MANIFEST_NAME
    if cached and manifest_path.is_file() and (INT8_DIR / "modules.json").is_file():
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(saved.get(k) == value for k, value in expected.items()) and saved.get("artifact_sha256") == _artifact_digest(cached):
                report = inspect_weightonly_graph(cached)
                log(f"Reusing validated weight-only INT8 graph: {report}")
                return cached, dict(expected, graph=report, preflight=preflight, reused=True)
        except Exception as exc:
            log(f"Cached weight-only graph rejected; rebuilding: {exc}")

    BASE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dinghy-weightonly-", dir=BASE))
    try:
        external = _onnx_external_paths(source_graph)
        shutil.copytree(FP32_DIR, staging, dirs_exist_ok=True, copy_function=shutil.copy2,
                        ignore=_copytree_ignore_onnx_source(source_graph, external))
        relative = source_graph.relative_to(FP32_DIR)
        destination = staging / relative.parent / f"{source_graph.stem}_{INT8_SUFFIX}{source_graph.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        log(f"Weight-only INT8: symmetric 8-bit MatMul weights | block={WEIGHT_BLOCK_SIZE} | FP32 activations | accuracy_level={WEIGHT_ACCURACY_LEVEL}")
        log(f"Source ONNX bundle={_artifact_size_bytes(source_graph) / (1024**3):.2f} GiB; external files={len(external)}")
        _run_weightonly_worker(source_graph, destination)
        report = inspect_weightonly_graph(destination)
        if report["matmul_nbits_nodes"] < 8:
            raise RuntimeError(f"Only {report['matmul_nbits_nodes']} MatMul nodes quantized; refusing an ineffective INT8 artifact")
        manifest = dict(expected, artifact_sha256=_artifact_digest(destination), graph=report,
                        preflight=preflight, artifact_size_bytes=_artifact_size_bytes(destination))
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        atomic_replace(staging, INT8_DIR)
        staging = None
        graph = find_int8_graph(INT8_DIR)
        if graph is None:
            raise RuntimeError("Installed weight-only INT8 graph is missing")
        log(f"PASS weight-only INT8 structure: {report}")
        return graph, dict(expected, graph=report, preflight=preflight, reused=False)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def short_metrics(stats):
    return {k: round(float(stats[k]), 5) for k in ("min", "mean", "p05")}


def compare_embedding(summary, key, reference, candidate):
    stats = cosine_stats(reference, candidate)
    summary["cosine"][key] = short_metrics(stats)
    passed = stats["min"] >= MIN_COSINE and stats["mean"] >= MIN_MEAN_COSINE
    summary["checks"].append(("PASS " if passed else "FAIL ") + key)
    log(f"{'PASS' if passed else 'FAIL'} {key}: {summary['cosine'][key]}")
    if not passed:
        raise RuntimeError(
            f"{key} fidelity failed: min={stats['min']:.6f} (required {MIN_COSINE}), "
            f"mean={stats['mean']:.6f} (required {MIN_MEAN_COSINE})"
        )


def compare_retrieval(summary, key, reference, candidate):
    result = retrieval_parity(reference, candidate)
    summary["retrieval"][key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}
    passed = (result["recall1_equal"] and result["recall5_equal"]
              and result["relevant_top1_equal"] and result["relevant_hits_equal"]
              and result["top1"] >= 0.99
              and result["jaccard5"] >= MIN_TOP5_JACCARD and result["mrr_drop"] <= MAX_MRR_DROP)
    summary["checks"].append(("PASS " if passed else "FAIL ") + key + " retrieval")
    log(f"{'PASS' if passed else 'FAIL'} {key} retrieval: {summary['retrieval'][key]}")
    if not passed:
        raise RuntimeError(f"{key} retrieval parity failed")


def size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 1)


def write_summary(summary):
    """Atomically write a human-readable JSON document (never JSON Lines)."""
    temp = SUMMARY_FILE.with_suffix(".json.tmp")
    payload = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, SUMMARY_FILE)


def main():
    started = time.perf_counter()
    digest = sha256_file(Path(__file__).resolve())
    log(f"SCRIPT: {Path(__file__).resolve()} | sha256={digest}")
    summary = {"status": "RUNNING", "model": MODEL_ID,
               "fixture": "synthetic parity: 4 queries / 14 passages; not a benchmark",
               "report_format": "pretty-printed JSON",
               "checks": [], "cosine": {}, "retrieval": {}, "files_mb": {}}
    stage = "environment"
    try:
        torch, ort, cpu = check_environment(summary)
        write_summary(summary)
        stage = "PyTorch reference"
        base, base_info = evaluate_variant("pytorch_fp32", MODEL_ID, "torch", ort, torch)
        summary["retrieval"]["pytorch"] = {k: round(v, 4) for k, v in retrieval_stats(base).items()}
        write_summary(summary)

        summary["speed"] = {"torch_rows_s": round(base_info["rows_per_second"], 2)}
        stage = "FP32 ONNX export"
        fp32_graph = prepare_fp32(ort)
        summary["files_mb"]["fp32_model_bundle"] = round(_artifact_size_bytes(fp32_graph) / (1024 * 1024), 1)
        stage = "FP32 ONNX validation"
        fp32_file = fp32_graph.relative_to(FP32_DIR).as_posix()
        fp32, fp32_info = evaluate_variant("onnx_fp32", FP32_DIR, "onnx", ort, torch, fp32_graph, fp32_file)
        if fp32_info["prompts"] != base_info["prompts"]:
            raise RuntimeError("FP32 prompt metadata differs from PyTorch")
        if fp32_info["tokenizer"]["sha256"] != base_info["tokenizer"]["sha256"]:
            raise RuntimeError("FP32 tokenizer output differs from PyTorch")
        summary["speed"]["fp32_rows_s"] = round(fp32_info["rows_per_second"], 2)
        summary["speed"]["fp32_vs_torch"] = round(fp32_info["rows_per_second"] / max(base_info["rows_per_second"], 1e-9), 3)
        summary["retrieval"]["fp32"] = {k: round(v, 4) for k, v in retrieval_stats(fp32).items()}
        compare_embedding(summary, "torch_fp32_query", base[0], fp32[0])
        compare_embedding(summary, "torch_fp32_document", base[1], fp32[1])
        compare_retrieval(summary, "torch_fp32", base, fp32)
        summary["checks"].append("PASS FP32 tokenizer and prompt parity")
        write_summary(summary)

        stage = "INT8 weight-only MatMulNBits quantization"
        summary["quantization"] = {"scheme": "RTN weight-only MatMulNBits",
                                   "scope": "constant MatMul weights only; other weights and operations remain FP32",
                                   "weight_bits": WEIGHT_BITS, "block_size": WEIGHT_BLOCK_SIZE,
                                   "symmetric": True, "accuracy_level": WEIGHT_ACCURACY_LEVEL,
                                   "activation_compute": "FP32", "minimum_cosine": MIN_COSINE,
                                   "minimum_mean_cosine": MIN_MEAN_COSINE,
                                   "format": "QOperator", "ops": ["MatMul"]}
        summary["memory"] = _cgroup_memory()
        write_summary(summary)
        int8_graph, qconfig = prepare_int8(ort, cpu)
        summary["files_mb"]["int8_model_bundle"] = round(_artifact_size_bytes(int8_graph) / (1024 * 1024), 1)
        summary["files_mb"]["int8_vs_fp32"] = round(_artifact_size_bytes(int8_graph) / max(_artifact_size_bytes(fp32_graph), 1), 3)
        summary["quantization"] = qconfig
        stage = "INT8 weight-only ONNX validation"
        int8_file = int8_graph.relative_to(INT8_DIR).as_posix()
        int8, int8_info = evaluate_variant("onnx_int8", INT8_DIR, "onnx", ort, torch, int8_graph, int8_file)
        if int8_info["prompts"] != base_info["prompts"]:
            raise RuntimeError("INT8 prompt metadata differs from PyTorch")
        if int8_info["tokenizer"]["sha256"] != base_info["tokenizer"]["sha256"]:
            raise RuntimeError("INT8 tokenizer output differs from PyTorch")
        summary["speed"]["int8_rows_s"] = round(int8_info["rows_per_second"], 2)
        summary["speed"]["int8_vs_fp32"] = round(int8_info["rows_per_second"] / max(fp32_info["rows_per_second"], 1e-9), 3)
        summary["retrieval"]["int8"] = {k: round(v, 4) for k, v in retrieval_stats(int8).items()}
        compare_embedding(summary, "fp32_int8_query", fp32[0], int8[0])
        compare_embedding(summary, "fp32_int8_document", fp32[1], int8[1])
        compare_embedding(summary, "torch_int8_query", base[0], int8[0])
        compare_embedding(summary, "torch_int8_document", base[1], int8[1])
        compare_retrieval(summary, "fp32_int8", fp32, int8)
        compare_retrieval(summary, "torch_int8", base, int8)
        summary["checks"].append("PASS INT8 tokenizer and prompt parity")
        summary["status"] = "PASS"
        log("\nALL CHECKS PASSED")
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["failed_stage"] = stage
        summary["error"] = f"{type(exc).__name__}: {exc}"
        log(f"\nFAILED during {stage}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if "memory" in summary:
            summary["memory"]["after"] = _cgroup_memory()
        summary["seconds"] = round(time.perf_counter() - started, 1)
        try:
            write_summary(summary)
            log(f"Summary JSON: {SUMMARY_FILE}")
        except Exception as exc:
            log(f"Summary JSON write failed: {exc}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_weightonly_worker":
        try:
            _weightonly_quantize_worker(sys.argv[2], sys.argv[3])
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    elif len(sys.argv) == 1:
        raise SystemExit(main())
    else:
        print("This script takes no user arguments. Run: python3 convert_and_test.py", flush=True)
        raise SystemExit(2)
