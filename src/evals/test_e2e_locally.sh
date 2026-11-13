#!/usr/bin/env bash
# =================================================================
# src/evals/test_e2e_locally.sh
#
# One-command evaluation harness for the custom case-law suite.
# Builds a stratified gold set from the live database, runs the
# retriever against it, and reports Recall/NDCG/MRR with bootstrap
# confidence intervals plus failure-mode breakdowns.
#
# Usage examples:
#   # Standard evaluation (default RRF k from config)
#   bash src/evals/test_e2e_locally.sh
#
#   # Sweep RRF k = [5, 10, 20, 60] and pick the winner
#   bash src/evals/test_e2e_locally.sh --find-optimal-rrf-k
#
#   # Evaluate at top-k=5 with a fresh gold set
#   bash src/evals/test_e2e_locally.sh --top-k 5 --rebuild-gold
#
#   # Full-tier gold (50 HOLDING / 50 LOOKUP / 30 VALIDITY / 20 negative)
#   bash src/evals/test_e2e_locally.sh --tier full
#
#   # Exact-scan baseline (bypass HNSW, measure recall ceiling)
#   bash src/evals/test_e2e_locally.sh --exact-baseline
#
# Exit codes:
#   0 — evaluation completed and report written
#   1 — evaluation failed (see log tail)
#   2 — invalid arguments or missing prerequisites
# =================================================================

set -uo pipefail

# ── Resolve paths ─────────────────────────────────────────────────
# Script lives at: src/evals/test_e2e_locally.sh
# SRC_DIR must be: <repo>/src  (one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"

# ── Activate venv if not already active ───────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    for candidate in \
        "${SRC_DIR}/.venv/bin/activate" \
        "${REPO_ROOT}/.venv/bin/activate"; do
        if [[ -f "$candidate" ]]; then
            # shellcheck source=/dev/null
            source "$candidate"
            break
        fi
    done
fi

# ── Environment defaults ──────────────────────────────────────────
: "${DATABASE_URL:=postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag}"
: "${RETRIEVER_MODEL_DIR:=${SRC_DIR}/indexing_pipeline/embedders}"
export DATABASE_URL RETRIEVER_MODEL_DIR
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python3}"

# ── Args ──────────────────────────────────────────────────────────
TOP_K=10
TIER="smoke"
SEED=20261106
FIND_RRF_K=0
REBUILD_GOLD=0
EXACT_BASELINE=0
PUBLISHED_ONLY=0
ALLOW_UNVERIFIED=1
VERBOSE=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top-k)              TOP_K="$2"; shift 2 ;;
        --tier)               TIER="$2"; shift 2 ;;
        --seed)               SEED="$2"; shift 2 ;;
        --find-optimal-rrf-k) FIND_RRF_K=1; shift ;;
        --rebuild-gold)       REBUILD_GOLD=1; shift ;;
        --exact-baseline)     EXACT_BASELINE=1; shift ;;
        --published-only)     PUBLISHED_ONLY=1; shift ;;
        --no-unverified)      ALLOW_UNVERIFIED=0; shift ;;
        --verbose)            VERBOSE=1; shift ;;
        --help|-h)            usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || (( TOP_K < 1 )); then
    echo "error: --top-k must be a positive integer" >&2; exit 2
fi
case "$TIER" in smoke|full) ;; *)
    echo "error: --tier must be smoke or full" >&2; exit 2 ;;
esac

# ── Derived paths ─────────────────────────────────────────────────
GOLD_DIR="${SRC_DIR}/evals/gold"
CUSTOM_GOLD="${GOLD_DIR}/custom_${TIER}.jsonl"
RESULTS_DIR="${REPO_ROOT}/results/$(date -u +%Y-%m-%d)"
mkdir -p "$GOLD_DIR" "$RESULTS_DIR"

cd "$SRC_DIR"

# ── Preflight ─────────────────────────────────────────────────────
preflight() {
    echo "================================================================"
    echo "  Retrieval eval — local acceptance harness"
    echo "================================================================"
    printf "  repo:       %s\n" "$REPO_ROOT"
    printf "  python:     %s\n" "$($PYTHON --version 2>&1)"
    printf "  dsn:        %s\n" "${DATABASE_URL##*@}"
    printf "  model:      %s\n" "$RETRIEVER_MODEL_DIR"
    printf "  top-k:      %s   tier: %s   seed: %s\n" "$TOP_K" "$TIER" "$SEED"
    (( FIND_RRF_K ))     && printf "  mode:       RRF K SWEEP [5, 10, 20, 60]\n"
    (( EXACT_BASELINE )) && printf "  mode:       EXACT SCAN BASELINE\n"
    (( PUBLISHED_ONLY )) && printf "  filter:     published_only=True\n"
    echo ""

    [[ -d "$RETRIEVER_MODEL_DIR" ]] || {
        echo "error: model dir not found: $RETRIEVER_MODEL_DIR" >&2; exit 2; }

    $PYTHON -c "import psycopg, numpy, retriever, evals" 2>/dev/null || {
        echo "error: python deps not importable (psycopg, numpy, retriever, evals)" >&2
        exit 2; }

    local n
    n=$($PYTHON - <<'PY'
import os, psycopg
try:
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=5) as c:
        print(c.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
except Exception:
    print(0)
PY
)
    if [[ -z "$n" ]] || (( n == 0 )); then
        echo "error: database unreachable or cases table empty" >&2; exit 2
    fi
    printf "  corpus:     %s cases loaded\n\n" "$n"
}

# ── Gold set building ─────────────────────────────────────────────
build_gold_official() {
    $PYTHON -m evals.custom_eval_builder \
        --source evals.source:build_source \
        --tier "$TIER" \
        --seed "$SEED" \
        --min-cases 500 \
        --out "$CUSTOM_GOLD"
}

build_gold_fallback() {
    echo "  official builder failed (empty pool at this scale?)"
    echo "  falling back: build whatever pools are non-empty..."
    $PYTHON - "$CUSTOM_GOLD" "$TIER" <<'GOLD_EOF'
import json, sys
from pathlib import Path

from evals.source import build_source

out_path = Path(sys.argv[1])
tier = sys.argv[2]
per_class = 3 if tier == "smoke" else 30

source = build_source()
try:
    fp = source.corpus_fingerprint()
    candidates = source.load_candidates()
finally:
    source.close()

pools = {"HOLDING": [], "LOOKUP": [], "VALIDITY": [], "negative_control": []}
for c in candidates:
    if c.get("holding_chunk_id") and (c.get("holding_text") or "").strip():
        pools["HOLDING"].append(c)
    if c["indegree"] >= 1 and c.get("citer_case_ids"):
        pools["LOOKUP"].append(c)
    if c["indegree"] >= 1 and c.get("overruling_citer_ids"):
        pools["VALIDITY"].append(c)
    if c["indegree"] == 0:
        pools["negative_control"].append(c)

gold = []
for label, pool in pools.items():
    take = min(per_class, len(pool))
    if take == 0:
        print(f"  skip {label}: empty pool", file=sys.stderr)
        continue
    for c in pool[:take]:
        cid = c["case_id"]
        if label == "HOLDING":
            q = f"What did the court hold in {c['case_name']}?"
            gc, gk = [cid], [c["holding_chunk_id"]]
        elif label == "LOOKUP":
            q = f"Which cases cite {c['case_name']}?"
            gc, gk = list(c["citer_case_ids"]), []
        elif label == "VALIDITY":
            q = f"Has {c['case_name']} been overruled?"
            gc, gk = list(c["overruling_citer_ids"]), []
        else:
            q = f"Which cases cite {c['case_name']}?"
            gc, gk = [], []
        gold.append({
            "id": f"{label.lower()}-{cid}",
            "suite": "custom",
            "query": q,
            "query_class_expected": label,
            "answerable": label != "negative_control",
            "gold_case_ids": gc,
            "gold_chunk_ids": gk,
            "seed_case_id": cid,
            "verified": False,
            "corpus_hash": fp,
        })

if not gold:
    print("FATAL: no gold rows produced from any pool", file=sys.stderr)
    sys.exit(1)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(json.dumps(r) for r in gold) + "\n")
print(f"  wrote {len(gold)} fallback gold rows -> {out_path}")
GOLD_EOF
}

ensure_gold() {
    if (( REBUILD_GOLD )) && [[ -f "$CUSTOM_GOLD" ]]; then
        echo "─── removing existing gold (--rebuild-gold) ───"
        rm -f "$CUSTOM_GOLD" "${CUSTOM_GOLD%.jsonl}.manifest.json"
    fi

    if [[ -f "$CUSTOM_GOLD" ]]; then
        echo "─── gold set exists: $CUSTOM_GOLD ───"
        return 0
    fi

    echo "─── building custom gold (tier=$TIER) ───"
    if build_gold_official; then
        return 0
    fi
    build_gold_fallback || {
        echo "FATAL: could not build gold set" >&2; exit 1; }
}

# ── Eval runner ───────────────────────────────────────────────────
run_one_eval() {
    local rrf_k="$1" out_file="$2" log_file="$3"

    local extra=()
    (( ALLOW_UNVERIFIED )) && extra+=("--allow-unverified")
    (( EXACT_BASELINE ))   && extra+=("--exact-baseline")
    (( PUBLISHED_ONLY ))   && extra+=("--published-only")
    [[ -n "$rrf_k" ]]      && extra+=("--rrf-k" "$rrf_k")

    $PYTHON -m evals.run_eval \
        --gold "$CUSTOM_GOLD" \
        --adapter evals.adapter:build_runtime \
        --suite custom \
        --config "e2e_${TIER}" \
        --top-k "$TOP_K" \
        --out "$out_file" \
        "${extra[@]}" > "$log_file" 2>&1
}

# ── Summary: single run ───────────────────────────────────────────
summarize_single() {
    $PYTHON - "$1" "$TOP_K" <<'SUM_EOF'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
k = int(sys.argv[2])
if not path.is_file():
    print("no report found"); sys.exit(1)
r = json.loads(path.read_text())

def fmt(mean, ci):
    if mean is None: return "—"
    if ci and len(ci) == 2:
        return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{mean:.3f}"

print()
print("=" * 78)
print(f"EVALUATION SUMMARY  (top_k={k}, n_queries={r['n_queries']})")
print("=" * 78)
print(f"  {'class':<14} {'N':>3}  {'recall@5':<22} {'recall@'+str(k):<22} {'ndcg@'+str(k):<8} {'mrr':<8}")
print(f"  {'-'*14} {'-'*3}  {'-'*22} {'-'*22} {'-'*8} {'-'*8}")

for cls, m in sorted(r.get("per_class", {}).items()):
    rec5 = m.get("case_recall@5")
    rec5_ci = m.get("case_recall@5_ci")
    rec_k = m.get(f"case_recall@{k}")
    rec_k_ci = m.get(f"case_recall@{k}_ci")
    nd = m.get(f"case_ndcg@{k}")
    mr = m.get("case_mrr")

    r5_s = fmt(rec5, rec5_ci)
    rk_s = fmt(rec_k, rec_k_ci)
    nd_s = f"{nd:.3f}" if nd is not None else "—"
    mr_s = f"{mr:.3f}" if mr is not None else "—"
    print(f"  {cls:<14} {m['n']:>3}  {r5_s:<22} {rk_s:<22} {nd_s:<8} {mr_s:<8}")

fm = r.get("failure_modes", {})
if fm:
    print()
    print("  Failure modes (failed queries only):")
    for cls, modes in sorted(fm.items()):
        if not modes: continue
        parts = ", ".join(f"{mode}={n}" for mode, n in modes.items() if mode != "success")
        if parts:
            print(f"    {cls:<18} {parts}")

lat = r.get("latency", {})
if lat.get("p95_ms") is not None:
    print()
    print(f"  Latency: p50={lat['p50_ms']:.0f}ms  p95={lat['p95_ms']:.0f}ms  mean={lat['mean_ms']:.0f}ms")

macro = r.get("macro_positive", {})
mr = macro.get(f"case_recall@{k}")
if mr is not None:
    print(f"  Macro recall@{k} (positive classes): {mr:.3f}")
print("=" * 78)
SUM_EOF
}

# ── Summary: RRF k sweep ──────────────────────────────────────────
summarize_sweep() {
    $PYTHON - "$RESULTS_DIR" "$TOP_K" <<'SWEEP_EOF'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
k_top = int(sys.argv[2])
ks = [5, 10, 20, 60]

print()
print("=" * 78)
print(f"RRF K SWEEP RESULTS  (top_k={k_top}, macro over positive classes)")
print("=" * 78)
print(f"  {'rrf_k':>5} | {'recall@k [95% CI]':<26} | {'ndcg@k':>7} | {'p95 lat':>8}")
print(f"  {'-'*5}-+-{'-'*26}-+-{'-'*7}-+-{'-'*8}")

rows = []
for k in ks:
    f = results_dir / f"eval_rrf_k{k}.json"
    if not f.is_file():
        print(f"  {k:>5} | {'MISSING':<26} | {'—':>7} | {'—':>8}")
        continue
    try:
        r = json.loads(f.read_text())
    except Exception:
        print(f"  {k:>5} | {'PARSE ERROR':<26} | {'—':>7} | {'—':>8}")
        continue

    per = r.get("per_class", {})
    recalls, ndcgs = [], []
    for cls, m in per.items():
        if cls == "negative_control": continue
        v = m.get(f"case_recall@{k_top}")
        if v is not None: recalls.append(v)
        v = m.get(f"case_ndcg@{k_top}")
        if v is not None: ndcgs.append(v)

    rec = sum(recalls)/len(recalls) if recalls else None
    nd = sum(ndcgs)/len(ndcgs) if ndcgs else None
    p95 = r.get("latency", {}).get("p95_ms")

    # crude CI over per-class means for display; per-query CIs live in the JSON
    rec_s = f"{rec:.3f}" if rec is not None else "—"
    nd_s = f"{nd:.3f}" if nd is not None else "—"
    p95_s = f"{p95:.0f}ms" if p95 is not None else "—"
    print(f"  {k:>5} | {rec_s:<26} | {nd_s:>7} | {p95_s:>8}")
    if rec is not None:
        rows.append((k, rec, nd))

print()
if rows:
    best = max(rows, key=lambda x: x[1])
    print(f"  BEST rrf_k = {best[0]}  (macro recall@{k_top} = {best[1]:.3f})")
    print()
    print("  NOTE: point estimates only. For significance, compare per-query")
    print("  bootstrap CIs in the individual eval_rrf_k*.json reports; if the")
    print("  CIs overlap, the difference is not statistically significant.")
else:
    print("  no valid results — check eval_rrf_k*.log for errors")
print("=" * 78)
SWEEP_EOF
}

# ── Main ──────────────────────────────────────────────────────────
preflight
ensure_gold

if (( FIND_RRF_K )); then
    echo "─── running RRF k sweep [5, 10, 20, 60] ───"
    any_ok=0
    for K in 5 10 20 60; do
        printf "  → rrf_k=%s ... " "$K"
        out="$RESULTS_DIR/eval_rrf_k${K}.json"
        log="$RESULTS_DIR/eval_rrf_k${K}.log"
        if run_one_eval "$K" "$out" "$log"; then
            echo "ok"
            any_ok=1
        else
            echo "FAILED"
            (( VERBOSE )) && tail -20 "$log" >&2
        fi
    done
    (( any_ok )) || { echo "all sweep runs failed" >&2; exit 1; }
    summarize_sweep
else
    echo "─── running custom eval ───"
    out="$RESULTS_DIR/eval_${TIER}.json"
    log="$RESULTS_DIR/eval_${TIER}.log"
    if run_one_eval "" "$out" "$log"; then
        summarize_single "$out"
        echo "  report: $out"
        echo "  log:    $log"
    else
        echo "eval failed; tail of $log:" >&2
        tail -25 "$log" >&2
        exit 1
    fi
fi

exit 0
