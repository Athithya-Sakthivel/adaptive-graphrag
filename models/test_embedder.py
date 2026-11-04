#!/usr/bin/env python3
"""
test_embedder.py

Download dinghy-law-0.6b-v1 (OpenVINO FP16) to tmp/embedder/,
validate embeddings are working, then benchmark batch_size and
max_seq_length configurations.

Run: python3 test_embedder.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_REPO = "Athithya117/dinghy-law-0.6b-v1-openvino-fp16"
LOCAL_DIR = Path("tmp/embedder")
EMB_DIM = 1024
EXPECTED_XML_SHA = "3398a4ca7c562afe7efe74b7514839c92ae4c5c4efb067cb3919b9ee2eef1c33"
EXPECTED_BIN_SHA = "9e66e5c37291f7e38c3689fa9ffd9d7755016f5f1594c00c2d0655d9c2fc4771"


def log(msg=""):
    print(msg, flush=True)


def fail(msg):
    log(f"FAIL: {msg}")
    sys.exit(1)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    bin_path = LOCAL_DIR / "openvino" / "openvino_model.bin"
    if bin_path.is_file() and bin_path.stat().st_size > 0:
        log(f"[skip] model already present at {LOCAL_DIR}")
        return
    log(f"[download] {MODEL_REPO} -> {LOCAL_DIR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.pyc", "__pycache__/**"],
    )


def verify_files() -> None:
    xml = LOCAL_DIR / "openvino" / "openvino_model.xml"
    binp = LOCAL_DIR / "openvino" / "openvino_model.bin"
    if not xml.is_file() or not binp.is_file():
        fail("OpenVINO IR files missing after download")
    xh, bh = sha256(xml), sha256(binp)
    if xh != EXPECTED_XML_SHA:
        fail(f"XML sha256 mismatch\n  expected {EXPECTED_XML_SHA}\n  got      {xh}")
    if bh != EXPECTED_BIN_SHA:
        fail(f"BIN sha256 mismatch\n  expected {EXPECTED_BIN_SHA}\n  got      {bh}")
    log(f"[ok] openvino_model.xml sha256={xh[:16]}...")
    log(f"[ok] openvino_model.bin sha256={bh[:16]}... ({binp.stat().st_size // (1024*1024)} MB)")


def load_model():
    from sentence_transformers import SentenceTransformer
    log(f"[load] {LOCAL_DIR} (openvino backend)")
    model = SentenceTransformer(
        str(LOCAL_DIR),
        backend="openvino",
        model_kwargs={
            "ov_config": {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(min(8, os.cpu_count() or 4)),
            }
        },
    )
    return model


def validate(model) -> None:
    log("\n[validate] embeddings")
    queries = [
        "When does the statute of limitations begin for a personal injury claim?",
        "What is the standard for summary judgment?",
    ]
    docs = [
        "For a personal injury claim, the statute of limitations generally begins to run when the cause of action accrues.",
        "Summary judgment is appropriate when there is no genuine dispute as to any material fact.",
        "The parol evidence rule may prevent a party from contradicting a fully integrated writing.",
    ]

    model.max_seq_length = 512
    q = model.encode_query(queries, normalize_embeddings=True, show_progress_bar=False)
    d = model.encode_document(docs, normalize_embeddings=True, show_progress_bar=False)

    if q.shape != (2, EMB_DIM):
        fail(f"query shape {q.shape}, expected (2, {EMB_DIM})")
    if d.shape != (3, EMB_DIM):
        fail(f"doc shape {d.shape}, expected (3, {EMB_DIM})")
    if not np.isfinite(q).all() or not np.isfinite(d).all():
        fail("non-finite embedding values")

    qn, dn = np.linalg.norm(q, axis=1), np.linalg.norm(d, axis=1)
    if np.max(np.abs(qn - 1.0)) > 1e-3:
        fail(f"query norms not 1: {qn}")
    if np.max(np.abs(dn - 1.0)) > 1e-3:
        fail(f"document norms not 1: {dn}")
    log(f"  query shape={q.shape} norms={qn.round(6).tolist()}")
    log(f"  doc   shape={d.shape} norms={dn.round(6).tolist()}")

    scores = q @ d.T
    if scores[0, 0] <= scores[0, 2]:
        fail(f"query0 should match doc0 over doc2: {scores[0].round(4).tolist()}")
    if scores[1, 1] <= scores[1, 0]:
        fail(f"query1 should match doc1 over doc0: {scores[1].round(4).tolist()}")
    log(f"  relevance matrix:\n{np.array2string(scores, precision=4, suppress_small=True)}")

    q_as_doc = model.encode_document(
        [queries[0]], normalize_embeddings=True, show_progress_bar=False
    )[0]
    sim = float(np.dot(q[0], q_as_doc))
    if sim > 0.99999:
        fail(f"query prefix appears ignored (cos={sim:.6f})")
    log(f"  query-vs-document prefix effect: cosine={sim:.4f} (<1.0)")
    log("[pass] embeddings validated")


def build_chunks(n_chunks: int = 500, tokens_per_chunk: int = 500) -> list[str]:
    base = (
        "The court held that the statute of limitations begins to run when the "
        "plaintiff discovers the injury, and equitable tolling may excuse a late "
        "filing only where extraordinary circumstances prevented timely action. "
    )
    target_chars = tokens_per_chunk * 4
    text = (base * ((target_chars // len(base)) + 1))[:target_chars]
    return [text] * n_chunks


def time_encode(model, texts, batch_size, sort=False):
    """Encode texts, optionally sorted by length to minimize batch padding."""
    tok = model.tokenizer
    if sort:
        lengths = [len(tok.encode(t)) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        sorted_texts = [texts[i] for i in order]
    else:
        sorted_texts = list(texts)
        order = None

    # Warm-up: pays OpenVINO compile once, so timing reflects inference only.
    model.encode_document(
        sorted_texts[:batch_size],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    start = time.perf_counter()
    vecs = model.encode_document(
        sorted_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    if order is not None:
        restored = np.empty_like(vecs)
        restored[order] = vecs
        vecs = restored

    return elapsed, vecs


def benchmark(model) -> None:
    log("\n[benchmark] 500 chunks x 500 tokens")
    chunks = build_chunks(n_chunks=500, tokens_per_chunk=500)

    configs = [
        ("A: max_len=8192, batch=2,  unsorted", 8192, 2, False),
        ("B: max_len=512,  batch=2,  unsorted", 512, 2, False),
        ("C: max_len=512,  batch=32, sorted", 512, 32, True),
    ]

    results = []
    for label, max_len, batch, sort in configs:
        model.max_seq_length = max_len
        elapsed, vecs = time_encode(model, chunks, batch, sort=sort)
        if vecs.shape != (500, EMB_DIM):
            fail(f"{label}: bad output shape {vecs.shape}")
        if not np.isfinite(vecs).all():
            fail(f"{label}: non-finite output")
        rps = len(chunks) / max(elapsed, 1e-9)
        results.append((label, elapsed, rps))
        log(f"  {label}: {elapsed:7.2f}s  {rps:7.2f} rows/s")

    base_rps = results[0][2]
    log("\n[summary]")
    log(f"  {'config':<38} {'elapsed':>9} {'rows/s':>9} {'speedup':>9}")
    log(f"  {'-'*38} {'-'*9} {'-'*9} {'-'*9}")
    for label, elapsed, rps in results:
        log(f"  {label:<38} {elapsed:>8.2f}s {rps:>9.2f} {rps/base_rps:>8.2f}x")

    if results[2][2] >= 3 * base_rps:
        log(f"\n  PASS: config C is {results[2][2]/base_rps:.1f}x faster than config A")
    else:
        log(f"\n  NOTE: config C speedup is {results[2][2]/base_rps:.2f}x on this hardware")
        log("        (larger speedups require Zen 4+ / VNNI / GPU)")


def main() -> None:
    log("=" * 60)
    log("dinghy-law embedder: download + validate + benchmark")
    log("=" * 60)
    download()
    verify_files()
    model = load_model()
    validate(model)
    benchmark(model)
    log("\nDONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\ninterrupted")
        sys.exit(130)
    except Exception as exc:
        log(f"\nUNEXPECTED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
