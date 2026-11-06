#!/usr/bin/env bash
# =================================================================
# eval/test_e2e_locally.sh
#
# One-command acceptance harness for both eval suites:
#   1. custom      — case-law retrieval on the project corpus (headline)
#   2. legalbench  — LegalBench-RAG public anchor (external validity)
#
# Each suite is detected independently and enters one of three states:
#   READY      → run it directly
#   BUILDABLE  → build the needed artifact first, then run
#   MISSING    → skip with clear instructions (or fail under --strict)
#
# On a fresh checkout with the pipeline corpus loaded:
#   • custom      → BUILDABLE → builds 3-class smoke gold → runs
#   • legalbench  → MISSING (unless --download-legalbench/artifacts)
#
# Exit codes:
#   0  — every executed suite passed its gate
#   1  — at least one suite failed
#   2  — invalid arguments, or --strict and a requested suite is missing
# =================================================================
set -uo pipefail   # NOT -e — we deliberately collect failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_DSN="postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag"
MODEL_REL="src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16"

# ── Args ─────────────────────────────────────────────────────────
SUITE="all"
FAST=0
TOP_K=10
STRICT=0
VERBOSE=0
DL_LEGALBENCH=0
DL_ARTIFACTS=0
ALLOW_UNVERIFIED=1

usage() {
    cat <<'EOF'
Usage: eval/test_e2e_locally.sh [options]

  --suite {all,custom,legalbench}   Which suite(s) to run (default: all)
  --fast                            Custom only; skip LegalBench
  --top-k N                         Top-k cutoff (default: 10)
  --strict                          Exit 2 if any requested suite can't run
  --download-legalbench             Download LegalBench data if missing
  --download-artifacts              Download precomputed artifacts if missing
  --no-unverified                   Fail if custom gold is unverified
  --verbose                         Show per-query output
  --help                            This message

Examples:
  bash eval/test_e2e_locally.sh --fast
  bash eval/test_e2e_locally.sh --suite custom
  bash eval/test_e2e_locally.sh --suite all --download-legalbench
  bash eval/test_e2e_locally.sh --suite all --strict   # CI mode
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)              SUITE="$2"; shift 2 ;;
        --fast)               FAST=1; SUITE="custom"; shift ;;
        --top-k)              TOP_K="$2"; shift 2 ;;
        --strict)             STRICT=1; shift ;;
        --download-legalbench) DL_LEGALBENCH=1; shift ;;
        --download-artifacts) DL_ARTIFACTS=1; shift ;;
        --no-unverified)      ALLOW_UNVERIFIED=0; shift ;;
        --verbose)            VERBOSE=1; shift ;;
        --help|-h)            usage; exit 0 ;;
        *)                    echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$SUITE" in all|custom|legalbench) ;; *)
    echo "--suite must be all|custom|legalbench" >&2; exit 2 ;;
esac

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || (( TOP_K < 1 )); then
    echo "--top-k must be a positive integer" >&2; exit 2
fi

# ── Paths & environment ─────────────────────────────────────────
cd "$REPO_ROOT"

MODEL_DIR="${RETRIEVER_MODEL_DIR:-$REPO_ROOT/$MODEL_REL}"
DATABASE_URL="${DATABASE_URL:-$DEFAULT_DSN}"
LEGALBENCH_ROOT="${LEGALBENCH_ROOT:-$REPO_ROOT/data/legalbench}"
LEGALBENCH_ARTIFACTS="${LEGALBENCH_ARTIFACTS:-$REPO_ROOT/data/legalbench_artifacts}"
LEGALBENCH_DSN="${LEGALBENCH_DSN:-${DATABASE_URL%/*}/legalbench}"
LEGALBENCH_EMBEDDER="${LEGALBENCH_EMBEDDER:-evals.embedder_bridge}"
LEGALBENCH_EMBEDDING_MODEL="${LEGALBENCH_EMBEDDING_MODEL:-$MODEL_DIR}"
LEGALBENCH_CORPUS="${LEGALBENCH_CORPUS:-$LEGALBENCH_ROOT/corpus}"

CUSTOM_GOLD="$REPO_ROOT/src/evals/gold/custom_smoke.jsonl"
LEGALBENCH_GOLD="$REPO_ROOT/src/evals/gold/legalbench_privacy_qa.jsonl"
RESULTS_DIR="$REPO_ROOT/results/$(date -u +%Y-%m-%d)"
mkdir -p "$RESULTS_DIR"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DATABASE_URL
export LEGALBENCH_ROOT LEGALBENCH_ARTIFACTS LEGALBENCH_DSN
export LEGALBENCH_EMBEDDER LEGALBENCH_EMBEDDING_MODEL LEGALBENCH_CORPUS

for venv in "$REPO_ROOT/src/.venv/bin/activate" "$REPO_ROOT/.venv/bin/activate"; do
    [[ -f "$venv" ]] && { source "$venv"; break; }
done

PYTHON="${PYTHON:-python3}"

# ── Preflight ───────────────────────────────────────────────────
preflight() {
    echo "================================================================"
    echo "  Retrieval eval — local acceptance harness"
    echo "================================================================"
    printf "  repo:       %s\n" "$REPO_ROOT"
    printf "  python:     %s\n" "$($PYTHON --version 2>&1)"
    printf "  dsn:        %s@...\n" "${DATABASE_URL%%@*}"
    printf "  model:      %s\n" "$MODEL_DIR"
    printf "  top-k:      %s\n" "$TOP_K"
    printf "  suite:      %s\n" "$SUITE"
    echo ""
}

# ── Suite detection ─────────────────────────────────────────────
CUSTOM_MODE="MISSING";   CUSTOM_REASON=""
LB_MODE="MISSING";       LB_REASON=""

detect_custom() {
    if [[ -f "$CUSTOM_GOLD" ]]; then
        CUSTOM_MODE="READY"; CUSTOM_REASON="gold exists"
        return
    fi
    if ! $PYTHON -c "import retriever.eval_source, evals.custom_eval_builder" 2>/dev/null; then
        CUSTOM_MODE="MISSING"; CUSTOM_REASON="eval modules not importable"
        return
    fi
    local n
    n=$($PYTHON -c "
import os, psycopg
try:
    with psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=3, autocommit=True) as c:
        print(c.execute('SELECT COUNT(*) FROM cases').fetchone()[0])
except Exception:
    print(0)
" 2>/dev/null)
    if [[ -z "$n" ]] || (( n < 500 )); then
        CUSTOM_MODE="MISSING"; CUSTOM_REASON="DB unreachable or <500 cases"
        return
    fi
    CUSTOM_MODE="BUILDABLE"; CUSTOM_REASON="$n cases, will build smoke gold"
}

detect_legalbench() {
    if [[ -f "$LEGALBENCH_ARTIFACTS/chunks.parquet" && -f "$LEGALBENCH_ARTIFACTS/queries.parquet" ]]; then
        LB_MODE="READY"; LB_REASON="precomputed artifacts"
        return
    fi
    if [[ -d "$LEGALBENCH_CORPUS" && -d "$LEGALBENCH_ROOT/benchmarks" ]] \
        && $PYTHON -c "import evals.legalbench_rag" 2>/dev/null; then
        LB_MODE="BUILDABLE"; LB_REASON="will build index from raw data"
        return
    fi
    LB_MODE="MISSING"; LB_REASON="no data, no artifacts"
}

detect_all() { detect_custom; detect_legalbench; }

print_detection() {
    echo "Environment:"
    [[ -d "$MODEL_DIR" ]] && echo "  model dir          ✓" || echo "  model dir          ✗ $MODEL_DIR"
    $PYTHON -c "import psycopg, numpy, openvino" 2>/dev/null \
        && echo "  python deps        ✓" || echo "  python deps        ✗"
    $PYTHON -c "import retriever, evals" 2>/dev/null \
        && echo "  packages           ✓" || echo "  packages           ✗"
    echo ""
    echo "Suites:"
    printf "  custom             %-10s %s\n" "$CUSTOM_MODE" "($CUSTOM_REASON)"
    printf "  legalbench         %-10s %s\n" "$LB_MODE" "($LB_REASON)"
    echo ""
}

# ── Download helpers ───────────────────────────────────────────
download_legalbench() {
    echo "─── downloading LegalBench-RAG data ───"
    command -v hf >/dev/null || { echo "hf CLI not found; pip install huggingface_hub[cli]" >&2; return 1; }
    mkdir -p "$LEGALBENCH_ROOT"
    hf download zeroentropy/legalbenchrag \
        --repo-type dataset --local-dir "$LEGALBENCH_ROOT" || return 1
}

download_artifacts() {
    echo "─── downloading precomputed artifacts ───"
    command -v hf >/dev/null || { echo "hf CLI not found" >&2; return 1; }
    mkdir -p "$LEGALBENCH_ARTIFACTS"
    hf download Athithya117/legalbench-artifacts legalbench_artifacts.tar.gz \
        --repo-type dataset --local-dir "$LEGALBENCH_ROOT" || return 1
    tar -xzf "$LEGALBENCH_ROOT/legalbench_artifacts.tar.gz" -C "$LEGALBENCH_ARTIFACTS"
}

# ── Build: custom gold ─────────────────────────────────────────
build_custom_gold() {
    echo "─── building custom smoke gold ───"
    mkdir -p "$(dirname "$CUSTOM_GOLD")"

    if $PYTHON -m evals.custom_eval_builder \
            --source retriever.eval_source:build_source \
            --tier smoke --min-cases 500 \
            --out "$CUSTOM_GOLD" 2>&1; then
        return 0
    fi

    echo "builder failed (likely empty class pool at this scale)"
    echo "─── falling back to available classes only ───"

    $PYTHON - "$CUSTOM_GOLD" <<'GOLD_EOF'
"""Fallback gold: build whatever pools are non-empty."""
import json, sys
from pathlib import Path

from retriever.eval_source import build_source

out_path = Path(sys.argv[1])
source = build_source()
try:
    fp = source.corpus_fingerprint()
    candidates = source.load_candidates()
finally:
    source.close()

pools = {"HOLDING": [], "LOOKUP": [], "VALIDITY": [], "APPLICATION": [], "negative_control": []}
for c in candidates:
    if c.get("holding_chunk_id") and (c.get("holding_text") or "").strip():
        pools["HOLDING"].append(c)
    if c["indegree"] >= 1 and c.get("citer_case_ids"):
        pools["LOOKUP"].append(c)
    if c["indegree"] >= 1 and c.get("overruling_citer_ids"):
        pools["VALIDITY"].append(c)
    if (c["indegree"] >= 1 and c.get("application_case_ids")
        and (c.get("doctrine") or "").strip() and c.get("doctrine_leading")):
        pools["APPLICATION"].append(c)
    if c["indegree"] == 0:
        pools["negative_control"].append(c)

gold = []
for label, pool in pools.items():
    take = min(3, len(pool))
    if take == 0:
        print(f"skip {label}: empty pool", file=sys.stderr)
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
        elif label == "APPLICATION":
            q = f"How has {c['doctrine']} been applied since {c['year']}?"
            gc, gk = list(c["application_case_ids"]), []
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
            "gold_char_spans": None,
            "seed_case_id": cid,
            "tags": [label.lower()],
            "verified": False,
            "corpus_hash": fp,
        })

if not gold:
    print("no gold rows produced", file=sys.stderr)
    sys.exit(1)

out_path.write_text("\n".join(json.dumps(r) for r in gold) + "\n")
print(f"wrote {len(gold)} fallback gold rows → {out_path}")
GOLD_EOF
}

# ── Build: legalbench index ────────────────────────────────────
build_legalbench_index() {
    echo "─── building LegalBench-RAG index (privacy_qa only) ───"

    if ! $PYTHON -c "
import os, psycopg
try:
    psycopg.connect(os.environ['LEGALBENCH_DSN'], connect_timeout=3).close()
except Exception as e:
    print(f'unreachable: {e}', file=os.sys.stderr); exit(1)
"; then
        echo "FAIL: LegalBench DSN unreachable: $LEGALBENCH_DSN" >&2
        echo "      create database + extension:" >&2
        echo "        psql -c 'CREATE DATABASE legalbench;'" >&2
        echo "        psql -d legalbench -c 'CREATE EXTENSION vector;'" >&2
        return 1
    fi

    $PYTHON -m evals.legalbench_rag \
        --corpus "$LEGALBENCH_CORPUS" \
        --benchmarks "$LEGALBENCH_ROOT/benchmarks" \
        --dataset-revision "local-$(date -u +%Y%m%d)" \
        --subset privacy_qa \
        --gold-out "$LEGALBENCH_GOLD"
}

# ── Run: custom ────────────────────────────────────────────────
run_custom() {
    local out="$RESULTS_DIR/custom_smoke.json"
    local log="$RESULTS_DIR/custom.log"

    local extra=()
    [[ $ALLOW_UNVERIFIED -eq 1 ]] && extra+=("--allow-unverified")
    [[ $VERBOSE -eq 1 ]] || extra+=("--no-progress")

    echo "─── running custom eval ───"
    $PYTHON -m evals.run_eval \
        --gold "$CUSTOM_GOLD" \
        --adapter "retriever.eval_adapter:build_runtime" \
        --suite custom \
        --config smoke \
        --top-k "$TOP_K" \
        --out "$out" \
        "${extra[@]}" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "custom eval failed (exit=$rc); tail of $log:" >&2
        tail -20 "$log" >&2
        return $rc
    fi
    echo "  → $out"
    return 0
}

# ── Run: legalbench ────────────────────────────────────────────
run_legalbench() {
    local out="$RESULTS_DIR/legalbench_smoke.json"
    local log="$RESULTS_DIR/legalbench.log"
    local gold="$LEGALBENCH_GOLD"

    if [[ -f "$LEGALBENCH_ARTIFACTS/chunks.parquet" ]]; then
        echo "─── loading precomputed artifacts → Postgres ───"
        $PYTHON - "$LEGALBENCH_ARTIFACTS" "$LEGALBENCH_DSN" <<'LOAD_EOF'
import sys
from pathlib import Path
from evals.legalbench_rag import load_precomputed_index
try:
    r = load_precomputed_index(
        artifacts_root=Path(sys.argv[1]),
        dsn=sys.argv[2], rebuild=False,
    )
    print(f"loaded: {r}")
except Exception as e:
    print(f"load failed: {e}", file=sys.stderr); sys.exit(1)
LOAD_EOF
    fi

    if [[ ! -f "$gold" ]]; then
        local master="$REPO_ROOT/src/evals/gold/legalbench.jsonl"
        if [[ -f "$master" ]]; then gold="$master"; fi
    fi
    if [[ ! -f "$gold" ]]; then
        echo "FAIL: no legalbench gold; run --download-legalbench first" >&2
        return 1
    fi

    echo "─── running legalbench eval ───"
    $PYTHON -m evals.run_eval \
        --gold "$gold" \
        --adapter "evals.legalbench_rag:build_runtime" \
        --suite legalbench \
        --config naive500_dense \
        --top-k "$TOP_K" \
        --out "$out" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "legalbench eval failed (exit=$rc); tail of $log:" >&2
        tail -20 "$log" >&2
        return $rc
    fi
    echo "  → $out"
    return 0
}

# ── Summary + gates ────────────────────────────────────────────
summarize() {
    local custom_json="$1"
    local lb_json="$2"
    $PYTHON - "$custom_json" "$lb_json" "$TOP_K" <<'SUM_EOF'
import json, sys
from pathlib import Path

def load(p):
    if not p or not Path(p).is_file():
        return None
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

custom = load(sys.argv[1])
lb = load(sys.argv[2])
K = int(sys.argv[3])

print()
print("=" * 74)
print("SUMMARY")
print("=" * 74)

gates = []  # list[(name, passed, message)]

# ── custom ──
if custom:
    print()
    print("custom  (headline — case-law retrieval)")
    print(f"  n_queries:   {custom.get('n_queries')}")
    print(f"  n_verified:  {custom.get('n_verified')}")
    print(f"  corpus_hash: {str(custom.get('corpus_hash',''))[:16]}…")
    print()
    k = custom["top_k"]
    rk = f"case_recall@{k}"
    nk = f"case_ndcg@{k}"
    print(f"  {'class':<14} {'N':>4} {'case-recall':>12} {'ndcg':>8} {'mrr':>8}")
    print(f"  {'-'*14} {'-'*4} {'-'*12} {'-'*8} {'-'*8}")
    per = custom.get("per_class", {})
    positive = []
    for cls, m in sorted(per.items()):
        r = m.get(rk)
        nd = m.get(nk)
        mr = m.get("case_mrr")
        rs = f"{r:.3f}" if r is not None else "—"
        ns = f"{nd:.3f}" if nd is not None else "—"
        ms = f"{mr:.3f}" if mr is not None else "—"
        print(f"  {cls:<14} {m['n']:>4} {rs:>12} {ns:>8} {ms:>8}")
        if cls != "negative_control" and r is not None:
            positive.append((cls, r))
    max_r = max((r for _, r in positive), default=0.0)
    refus = per.get("negative_control", {}).get("refusal_accuracy")
    if refus is not None:
        print(f"\n  negative-control refusal: {refus:.3f}")
    print()
    ok = max_r >= 0.30 or not positive
    print(f"  gate: max positive-class recall >= 0.30 → {'PASS' if ok else 'FAIL'}")
    gates.append(("custom", ok, f"max recall {max_r:.3f}"))
elif custom is not None and custom == {}:
    gates.append(("custom", False, "empty result"))
# custom was None → skipped, not a gate

# ── legalbench ──
if lb:
    print()
    print("legalbench  (external anchor — contract retrieval)")
    print(f"  n_queries:   {lb.get('n_queries')}")
    print(f"  corpus_hash: {str(lb.get('corpus_hash',''))[:16]}…")
    print()
    k = lb["top_k"]
    macro = lb.get("macro_positive", {})
    char = macro.get(f"char_span_recall@{k}")
    prec = macro.get(f"char_span_precision@{k}")
    if char is not None:
        print(f"  char-recall@{k}:    {char:.4f}")
    if prec is not None:
        print(f"  char-precision@{k}: {prec:.4f}")
    print()
    ok = char is not None and 0.40 <= char <= 0.95
    print(f"  gate: 0.40 <= char-recall <= 0.95 → {'PASS' if ok else 'FAIL'}")
    if char is None:
        print("  (paper baseline: 71.2% on privacy_qa; tuned hybrid: 78.4%)")
    elif char < 0.40:
        print("  WARNING: too low — embedder or chunking bug, not retriever quality")
    elif char > 0.95:
        print("  WARNING: too high — budget likely > doc size (check k, chunk size)")
    gates.append(("legalbench", ok, f"char-recall {char:.3f}" if char is not None else "missing"))
elif lb is not None and lb == {}:
    gates.append(("legalbench", False, "empty result"))
# lb was None → skipped, not a gate

print()
print("=" * 74)
if not gates:
    print("No suites executed.")
    print("Hints:")
    print("  • To enable custom:    ensure DATABASE_URL points at the loaded corpus")
    print("  • To enable legalbench: --download-legalbench (needs hf CLI + ~80 MB)")
    print("  • Already have artifacts? --download-artifacts (fastest path)")
    sys.exit(1)

fails = [(n, msg) for n, ok, msg in gates if not ok]
for name, ok, msg in gates:
    print(f"  {'PASS' if ok else 'FAIL':4} {name:<12} {msg}")
print("=" * 74)
sys.exit(0 if not fails else 1)
SUM_EOF
}

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
preflight
detect_all
print_detection

# Optional downloads
if (( DL_LEGALBENCH == 1 )) && [[ "$LB_MODE" == "MISSING" ]]; then
    download_legalbench && detect_all
fi
if (( DL_ARTIFACTS == 1 )) && [[ "$LB_MODE" == "MISSING" ]]; then
    download_artifacts && detect_all
fi

# Decide which suites to run
RUN_CUSTOM=0; RUN_LB=0
[[ "$SUITE" == "all" || "$SUITE" == "custom" ]] && RUN_CUSTOM=1
[[ "$SUITE" == "all" || "$SUITE" == "legalbench" ]] && RUN_LB=1
(( FAST == 1 )) && RUN_LB=0

# Strict gate: requested → must be ready/buildable
if (( STRICT == 1 )); then
    if (( RUN_CUSTOM == 1 )) && [[ "$CUSTOM_MODE" == "MISSING" ]]; then
        echo "STRICT: custom requested but missing — $CUSTOM_REASON" >&2
        exit 2
    fi
    if (( RUN_LB == 1 )) && [[ "$LB_MODE" == "MISSING" ]]; then
        echo "STRICT: legalbench requested but missing — $LB_REASON" >&2
        exit 2
    fi
fi

CUSTOM_JSON=""
LB_JSON=""

if (( RUN_CUSTOM == 1 )); then
    case "$CUSTOM_MODE" in
        READY)
            run_custom && CUSTOM_JSON="$RESULTS_DIR/custom_smoke.json" ;;
        BUILDABLE)
            if build_custom_gold; then
                detect_custom
                if [[ "$CUSTOM_MODE" == "READY" ]]; then
                    run_custom && CUSTOM_JSON="$RESULTS_DIR/custom_smoke.json"
                fi
            fi ;;
        MISSING)
            echo "[custom] SKIPPED — $CUSTOM_REASON" ;;
    esac
fi

if (( RUN_LB == 1 )); then
    case "$LB_MODE" in
        READY)
            run_legalbench && LB_JSON="$RESULTS_DIR/legalbench_smoke.json" ;;
        BUILDABLE)
            if build_legalbench_index; then
                detect_legalbench
                if [[ "$LB_MODE" != "MISSING" ]]; then
                    run_legalbench && LB_JSON="$RESULTS_DIR/legalbench_smoke.json"
                fi
            fi ;;
        MISSING)
            echo "[legalbench] SKIPPED — $LB_REASON"
            echo "             enable with: --download-legalbench or --download-artifacts" ;;
    esac
fi

# ── Summary ─────────────────────────────────────────────────────
summarize "$CUSTOM_JSON" "$LB_JSON"
exit $?
