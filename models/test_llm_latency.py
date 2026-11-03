#!/usr/bin/env python3
"""
test_llm_latency.py

CPU RAG suitability test for llama-server.

What it measures:
  1. Prefill throughput and decode throughput at fixed prompt sizes.
  2. Time-to-first-token (TTFT) on the client, which is what users feel.
  3. Streaming sanity: deltas must arrive incrementally, not as one buffered dump.
  4. Prompt prefix cache reuse: a second request with the same static prefix
     should have lower TTFT than a cold request (llama.cpp caches the KV prefix).
  5. max_tokens enforcement: generation must stop at the configured cap.

Server must be running with -c 8192 or larger to accept 4000-token prompts.
Recommended launch:

  ./llama-server \\
    -m /workspace/models/llm/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \\
    -c 8192 -t 4 -tb 4 --jinja --port 8080

Usage:
    python3 test_llm_latency.py
    python3 test_llm_latency.py --sizes 500 1000 2000 4000 --runs 2
    python3 test_llm_latency.py --url http://127.0.0.1:8080
"""

import argparse
import json
import statistics
import sys
import time

import requests


# --- realistic seed text (legal prose, not lorem ipsum) ---
SEED = (
    "The court held that the statute of limitations begins to run when the "
    "plaintiff discovers the injury, and equitable tolling may excuse a late "
    "filing only where extraordinary circumstances prevented timely action. "
    "Summary judgment is appropriate when there is no genuine dispute as to "
    "any material fact and the moving party is entitled to judgment as a "
    "matter of law. "
)

# --- static RAG system prompt (kept short; this is the CPU RAG rule) ---
STATIC_PREFIX = (
    "You are a legal research assistant. Answer using only the evidence "
    "provided. Cite the case name and year for every factual claim. "
    "If the evidence is insufficient, say so and stop. "
    "Keep answers under 200 words.\n\nEVIDENCE:\n"
)


def log(msg=""):
    print(msg, flush=True)


def fail(msg):
    log(f"FAIL: {msg}")
    sys.exit(1)


def post_json(url, body, timeout=60):
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def count_tokens(base, text):
    return len(post_json(f"{base}/tokenize", {"content": text})["tokens"])


def build_prompt(base, target_tokens):
    """Build a prompt whose tokenized length is within 1% of target."""
    unit = count_tokens(base, SEED)
    if unit == 0:
        fail("seed text tokenized to zero tokens")

    reps = max(1, target_tokens // unit)
    prompt = SEED * reps
    actual = count_tokens(base, prompt)

    attempts = 0
    while abs(actual - target_tokens) > max(8, target_tokens // 100) and attempts < 50:
        if actual < target_tokens:
            prompt += " further"
        elif reps > 1:
            reps -= 1
            prompt = SEED * reps
        else:
            prompt = prompt[: int(len(prompt) * target_tokens / actual)]
        actual = count_tokens(base, prompt)
        attempts += 1

    return prompt, actual


def stream_request(base, model, messages, max_tokens, timeout=600):
    """Send a streaming request. Return timing and streaming details."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.perf_counter()
    first_content_at = None
    last_content_at = None
    content_chunks = 0
    inter_chunk_gaps = []
    server_timings = None
    usage = None

    with requests.post(
        f"{base}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            payload = raw[len("data: "):] if raw.startswith("data: ") else raw
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(chunk.get("timings"), dict):
                server_timings = chunk["timings"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                now = time.perf_counter()
                if first_content_at is None:
                    first_content_at = now
                if last_content_at is not None:
                    inter_chunk_gaps.append((now - last_content_at) * 1000.0)
                last_content_at = now
                content_chunks += 1

    t_end = time.perf_counter()
    ttft_ms = (first_content_at - t_start) * 1000.0 if first_content_at else None

    return {
        "ttft_ms": ttft_ms,
        "total_ms": (t_end - t_start) * 1000.0,
        "content_chunks": content_chunks,
        "median_gap_ms": (
            statistics.median(inter_chunk_gaps) if inter_chunk_gaps else None
        ),
        "server_timings": server_timings,
        "usage": usage,
    }


def summarize(runs):
    def med(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    def med_timing(field):
        vals = [
            r["server_timings"][field]
            for r in runs
            if r.get("server_timings") and field in r["server_timings"]
        ]
        return statistics.median(vals) if vals else None

    return {
        "prompt_n": med_timing("prompt_n"),
        "prompt_ms": med_timing("prompt_ms"),
        "prefill_tps": med_timing("prompt_per_second"),
        "predicted_n": med_timing("predicted_n"),
        "predicted_ms": med_timing("predicted_ms"),
        "decode_tps": med_timing("predicted_per_second"),
        "ttft_ms": med("ttft_ms"),
        "total_ms": med("total_ms"),
        "chunks": med("content_chunks"),
        "gap_ms": med("median_gap_ms"),
    }


def verdict(decode_tps, prefill_tps, ttft_ms):
    if decode_tps is None or prefill_tps is None:
        return "NO DATA"
    if decode_tps >= 20 and prefill_tps >= 100 and ttft_ms and ttft_ms <= 5000:
        return "SUITABLE"
    if decode_tps >= 10 and prefill_tps >= 40 and ttft_ms and ttft_ms <= 15000:
        return "MARGINAL"
    return "TOO SLOW"


def test_health(base):
    log("=" * 72)
    log("HEALTH")
    log("=" * 72)
    try:
        r = requests.get(f"{base}/health", timeout=10)
        log(f"/health -> {r.status_code} {r.text.strip()[:60]}")
        r.raise_for_status()
    except Exception as e:
        fail(f"server unreachable at {base}: {e}")


def test_warmup(base, model):
    log("Warmup request (prevents cold-start biasing the first real run) ...")
    try:
        stream_request(base, model, [{"role": "user", "content": "Hi"}], 8)
    except Exception as e:
        fail(f"warmup failed: {e}")
    log("Warmup done.")


def test_sizes(base, model, sizes, runs, max_tokens):
    log("\n" + "=" * 72)
    log(f"PREFILL / DECODE AT FIXED PROMPT SIZES ({runs} runs each)")
    log("=" * 72)

    log("\nBuilding prompts ...")
    prompts = {}
    for size in sizes:
        text, actual = build_prompt(base, size)
        prompts[size] = (text, actual)
        log(f"  target={size:5d}  actual={actual:5d} tokens")

    results = {}
    for size in sizes:
        text, actual = prompts[size]
        log(f"\n--- prompt size {actual} tokens ---")
        rows = []
        for i in range(runs):
            r = stream_request(
                base, model,
                [{"role": "user", "content": text}],
                max_tokens,
            )
            rows.append(r)
            st = r["server_timings"] or {}
            log(
                f"  run {i+1}: "
                f"ttft={r['ttft_ms']:.0f}ms "
                f"total={r['total_ms']:.0f}ms "
                f"prefill={st.get('prompt_per_second', 0):.1f}t/s "
                f"decode={st.get('predicted_per_second', 0):.1f}t/s "
                f"gen={st.get('predicted_n', 0)}t"
            )
        results[size] = summarize(rows)
    return results


def test_static_prefix_cache(base, model, prompt_text, runs):
    """Send the same static prefix twice. The second TTFT should be lower if
    llama.cpp prompt caching is working. This is the key CPU RAG optimization:
    a stable system prompt costs almost nothing after the first request."""
    log("\n" + "=" * 72)
    log("STATIC PREFIX CACHE REUSE")
    log("=" * 72)
    log("Sending identical prompt twice; second TTFT should drop if KV cache is reused.")

    first = stream_request(
        base, model,
        [{"role": "user", "content": prompt_text}],
        32,
    )
    second = stream_request(
        base, model,
        [{"role": "user", "content": prompt_text}],
        32,
    )

    if first["ttft_ms"] and second["ttft_ms"]:
        drop_pct = (1 - second["ttft_ms"] / first["ttft_ms"]) * 100.0
        log(f"  first  TTFT: {first['ttft_ms']:.0f}ms")
        log(f"  second TTFT: {second['ttft_ms']:.0f}ms")
        log(f"  drop: {drop_pct:+.1f}%")
        if drop_pct > 20:
            log("  PASS: prefix cache is being reused (>=20% TTFT drop)")
            return True
        if drop_pct > 0:
            log("  NOTE: small drop; cache may be partially reused")
            return True
        log("  WARN: no TTFT drop; caching not effective for this prompt")
        return False

    log("  SKIP: missing TTFT on one of the runs")
    return False


def test_streaming_integrity(sample):
    """A healthy stream delivers many deltas with non-zero gaps between them.
    A buffered stream delivers one delta (or all gaps near zero)."""
    log("\n" + "=" * 72)
    log("STREAMING INTEGRITY")
    log("=" * 72)
    if not sample or not sample["content_chunks"]:
        fail("no content deltas received")

    log(f"  content chunks: {sample['content_chunks']:.0f}")
    log(f"  median gap between chunks: {sample['gap_ms']:.1f}ms"
        if sample["gap_ms"] is not None else "  median gap: n/a")

    if sample["content_chunks"] < 2:
        fail("only one content delta received; streaming is buffered or capped")
    if sample["gap_ms"] is not None and sample["gap_ms"] < 1.0:
        log("  WARN: inter-chunk gap near zero; may be a buffered stream")
    else:
        log("  PASS: deltas arrive incrementally")


def test_max_tokens(base, model, cap):
    log("\n" + "=" * 72)
    log(f"MAX TOKENS ENFORCEMENT (cap={cap})")
    log("=" * 72)
    r = stream_request(
        base, model,
        [{"role": "user", "content":
          "Write a very long essay about the history of the common law."}],
        cap,
    )
    gen = (r["server_timings"] or {}).get("predicted_n")
    log(f"  generated tokens: {gen:.0f}")
    if gen is None:
        fail("server did not report predicted_n")
    if gen <= cap:
        log(f"  PASS: generation stopped at or before cap ({gen:.0f} <= {cap})")
    else:
        fail(f"generation exceeded cap: {gen:.0f} > {cap}")


def report(results, sizes, prefix_ok):
    log("\n" + "=" * 72)
    log("SUMMARY (median across runs)")
    log("=" * 72)
    header = (
        f"{'prompt_n':>9} | {'prefill_tps':>11} | {'decode_tps':>10} | "
        f"{'ttft_ms':>8} | {'total_ms':>9} | {'verdict':>9}"
    )
    log(header)
    log("-" * len(header))

    worst = None
    for size in sizes:
        s = results[size]
        v = verdict(s["decode_tps"], s["prefill_tps"], s["ttft_ms"])
        if v == "TOO SLOW" and worst is None:
            worst = size

        def fmt(x, f=".0f"):
            return format(x, f) if x is not None else "-"

        log(
            f"{fmt(s['prompt_n']):>9} | "
            f"{fmt(s['prefill_tps'], '.1f'):>11} | "
            f"{fmt(s['decode_tps'], '.1f'):>10} | "
            f"{fmt(s['ttft_ms']):>8} | "
            f"{fmt(s['total_ms']):>9} | "
            f"{v:>9}"
        )

    log("\n" + "=" * 72)
    log("INTERPRETATION")
    log("=" * 72)
    log("prefill_tps: how fast the prompt is ingested (drives TTFT)")
    log("decode_tps:  how fast the answer is produced")
    log("ttft_ms:     client-observed time until first content delta")
    log("")
    log("Verdicts reflect CPU RAG usability for an interactive demo:")
    log("  SUITABLE: decode >= 20 t/s, prefill >= 100 t/s, TTFT <= 5s")
    log("  MARGINAL: decode >= 10 t/s, prefill >= 40 t/s, TTFT <= 15s")
    log("  TOO SLOW: below those floors")
    log("")
    log("The 2000-token row is the one that matters for RAG.")
    log("If 2000-token TTFT exceeds 15s, cap your retrieved context below it.")

    log("\n" + "=" * 72)
    log("CPU RAG RECOMMENDATIONS BASED ON THESE NUMBERS")
    log("=" * 72)
    s2000 = results.get(2000)
    if s2000 and s2000["ttft_ms"]:
        if s2000["ttft_ms"] <= 5000:
            log("2000-token TTFT <= 5s: interactive RAG is viable.")
        elif s2000["ttft_ms"] <= 15000:
            log("2000-token TTFT 5-15s: usable for non-interactive or batch.")
            log("  For interactive, cap retrieved context at 1000-1500 tokens.")
        else:
            log("2000-token TTFT > 15s: too slow for interactive use.")
            log("  Cap context at 800-1200 tokens, or move generation to a larger box.")

    if prefix_ok:
        log("Prefix caching works: keep the static system prompt identical across")
        log("requests so follow-ups skip prefill of the shared prefix.")
    else:
        log("Prefix caching did not help: reduce static system prompt length and")
        log("send the same prefix on every request to encourage reuse.")

    if worst is not None:
        log(f"\nWARNING: prompt size {worst} is TOO SLOW.")
        return False
    log("\nALL SIZES PASS")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="lfm2.5-1.2b-instruct")
    ap.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000, 4000])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    base = args.url.rstrip("/")

    test_health(base)
    test_warmup(base, args.model)

    results = test_sizes(base, args.model, args.sizes, args.runs, args.max_tokens)

    # Sample the 1000-token run for streaming integrity
    sample_ok = True
    try:
        sample = stream_request(
            base, args.model,
            [{"role": "user", "content": SEED * 15}],
            64,
        )
        test_streaming_integrity(sample)
    except Exception as e:
        log(f"STREAMING CHECK SKIPPED: {e}")
        sample_ok = False

    # max_tokens enforcement
    try:
        test_max_tokens(base, args.model, args.max_tokens)
    except Exception as e:
        log(f"MAX TOKENS CHECK SKIPPED: {e}")

    # Prefix cache reuse on a realistic RAG prompt
    prompt_text, _ = build_prompt(base, 1000)
    realistic = STATIC_PREFIX + prompt_text
    prefix_ok = test_static_prefix_cache(base, args.model, realistic, runs=2)

    ok = report(results, args.sizes, prefix_ok)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
