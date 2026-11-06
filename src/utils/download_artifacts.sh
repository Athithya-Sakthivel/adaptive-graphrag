#!/usr/bin/env bash
# =================================================================
# src/utils/download_artifacts.sh
#
# Restore pinned artifacts from private HuggingFace datasets.
#
# Required env (exactly four, no aliases):
#   HF_USER_NAME       HuggingFace username
#   HF_TOKEN           read token (write token also works)
#   HF_LAW_REVISION    full 40-char SHA for legalrag-artifacts
#   HF_LB_REVISION     full 40-char SHA for legalbench-artifacts
#
# Usage:
#   HF_USER_NAME=Athithya117 \
#   HF_TOKEN=hf_... \
#   HF_LAW_REVISION=690800ee6c49f0bfde6c1e26f64031601feba6ae \
#   HF_LB_REVISION=513b8d9857fe13dde2af2c73f90dc0a0b063405b \
#     ./src/utils/download_artifacts.sh [options] [data_root]
#
# Options:
#   --components {all,case-law,legalbench}   default: all
#   --legalbench-dir PATH                    default: /workspace/data/legalbench_artifacts
#   --force                                  re-download even if extracted
#   --no-xet                                 force HTTP path (resumes partials)
#   --list-commits                           print commits per repo, exit
#   -h, --help                               show this message
# =================================================================


set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,/^# ===/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ── Args ───────────────────────────────────────────────────────────
COMPONENTS="all"
DATA_ROOT="tmp/data"
LEGALBENCH_DIR="${DATA_ROOT}/legalbench_artifacts"
FORCE=0
NO_XET=0
LIST_COMMITS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --components)       COMPONENTS="$2"; shift 2 ;;
        --components=*)     COMPONENTS="${1#*=}"; shift ;;
        --legalbench-dir)   LEGALBENCH_DIR="$2"; shift 2 ;;
        --legalbench-dir=*) LEGALBENCH_DIR="${1#*=}"; shift ;;
        --force)            FORCE=1; shift ;;
        --no-xet)           NO_XET=1; shift ;;
        --list-commits)     LIST_COMMITS=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; break ;;
        -*)                 echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                  DATA_ROOT="$1"; shift ;;
    esac
done

case "$COMPONENTS" in
    all|case-law|legalbench) ;;
    *) echo "--components must be all|case-law|legalbench" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-/workspace/tmp/data}"

# ── Required env ───────────────────────────────────────────────────
HF_USER_NAME="Athithya117"
: "${HF_LAW_REVISION:?HF_LAW_REVISION is required}"
: "${HF_LB_REVISION:?HF_LB_REVISION is required}"

# Full 40-char SHAs only — hf download rejects short hashes.
for pair in "HF_LAW_REVISION:$HF_LAW_REVISION" "HF_LB_REVISION:$HF_LB_REVISION"; do
    name="${pair%%:*}"; val="${pair#*:}"
    if (( ${#val} != 40 )); then
        echo "error: $name='$val' is not a 40-char SHA" >&2
        echo "       get it from: hf download --help | grep revision" >&2
        echo "       or run: --list-commits" >&2
        exit 2
    fi
done

REPO_LAW="${HF_USER_NAME}/legalrag-artifacts"
REPO_LB="${HF_USER_NAME}/legalbench-artifacts"

# ── Preflight ──────────────────────────────────────────────────────
command -v hf >/dev/null 2>&1 || {
    echo "error: hf CLI not found; pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
}

if (( NO_XET == 1 )); then
    export HF_HUB_DISABLE_XET=1
    echo "[env] HF_HUB_DISABLE_XET=1 (HTTP path — resumes partials)"
fi

# ── Diagnostic: list commits ───────────────────────────────────────
if (( LIST_COMMITS == 1 )); then
    python3 - "$REPO_LAW" "$REPO_LB" <<'PYEOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
for repo in sys.argv[1:]:
    print(f"\n{repo}")
    try:
        for c in api.list_repo_commits(repo, repo_type="dataset"):
            print(f"  {c.commit_id}  {c.created_at:%Y-%m-%d %H:%M}  {c.title}")
    except Exception as e:
        print(f"  (failed: {e})")
PYEOF
    exit 0
fi

# ── Helpers ────────────────────────────────────────────────────────
hdr() { printf "\n════ %s ════\n" "$*"; }
sub() { printf "  %s\n" "$*"; }

# ── case-law ───────────────────────────────────────────────────────
restore_case_law() {
    hdr "case-law"
    sub "repo:     $REPO_LAW"
    sub "revision: $HF_LAW_REVISION"
    sub "target:   $DATA_ROOT"
    echo

    if (( FORCE == 0 )) \
        && [[ -f "$DATA_ROOT/chunked/_manifest.jsonl" ]] \
        && [[ -f "$DATA_ROOT/embedded/_manifest.jsonl" ]]; then
        sub "already extracted; skipping (--force to re-download)"
        return 0
    fi

    mkdir -p "$DATA_ROOT"

    sub "downloading tarballs..."
    hf download "$REPO_LAW" \
        --repo-type dataset \
        --revision "$HF_LAW_REVISION" \
        --include "chunked.tar.gz" \
        --include "embedded.tar.gz" \
        --local-dir "$DATA_ROOT"

    sub "extracting..."
    rm -rf "$DATA_ROOT/chunked" "$DATA_ROOT/embedded"
    tar -xzf "$DATA_ROOT/chunked.tar.gz"  -C "$DATA_ROOT"
    tar -xzf "$DATA_ROOT/embedded.tar.gz" -C "$DATA_ROOT"
    rm -f "$DATA_ROOT/chunked.tar.gz" "$DATA_ROOT/embedded.tar.gz"

    local c e
    c=$(find "$DATA_ROOT/chunked"  -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
    e=$(find "$DATA_ROOT/embedded" -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
    sub "extracted: chunked=$c embedded=$e"
    return 0
}

# ── legalbench ─────────────────────────────────────────────────────
restore_legalbench() {
    hdr "legalbench"
    sub "repo:     $REPO_LB"
    sub "revision: $HF_LB_REVISION"
    sub "target:   $LEGALBENCH_DIR"
    echo

    if (( FORCE == 0 )) \
        && [[ -f "$LEGALBENCH_DIR/chunks.parquet" ]] \
        && [[ -f "$LEGALBENCH_DIR/queries.parquet" ]] \
        && [[ -f "$LEGALBENCH_DIR/manifest.json" ]]; then
        sub "already extracted; skipping (--force to re-download)"
        return 0
    fi

    mkdir -p "$LEGALBENCH_DIR"

    sub "downloading tarball..."
    hf download "$REPO_LB" legalbench_artifacts.tar.gz \
        --repo-type dataset \
        --revision "$HF_LB_REVISION" \
        --local-dir "$LEGALBENCH_DIR"

    sub "extracting..."
    tar -xzf "$LEGALBENCH_DIR/legalbench_artifacts.tar.gz" -C "$LEGALBENCH_DIR"
    rm -f "$LEGALBENCH_DIR/legalbench_artifacts.tar.gz"

    for required in chunks.parquet queries.parquet manifest.json; do
        [[ -f "$LEGALBENCH_DIR/$required" ]] || {
            echo "  error: missing $required after extraction" >&2
            return 1
        }
    done

    sub "extracted: $(du -sh "$LEGALBENCH_DIR" | cut -f1)"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────
echo "================================================================"
echo "  Artifact download"
echo "================================================================"
printf "  components: %s\n" "$COMPONENTS"
printf "  HF user:    %s\n" "$HF_USER_NAME"

WANT_LAW=0; WANT_LB=0
[[ "$COMPONENTS" == "all" || "$COMPONENTS" == "case-law"  ]] && WANT_LAW=1
[[ "$COMPONENTS" == "all" || "$COMPONENTS" == "legalbench" ]] && WANT_LB=1

[[ $WANT_LAW -eq 1 ]] && printf "  case-law:   %s @ %s\n" "$REPO_LAW" "${HF_LAW_REVISION:0:12}…"
[[ $WANT_LB  -eq 1 ]] && printf "  legalbench: %s @ %s\n" "$REPO_LB"  "${HF_LB_REVISION:0:12}…"

FAILED=0
if [[ $WANT_LAW -eq 1 ]] && ! restore_case_law; then
    echo "case-law: FAILED" >&2; FAILED=1
fi
if [[ $WANT_LB -eq 1 ]] && ! restore_legalbench; then
    echo "legalbench: FAILED" >&2; FAILED=1
fi

# ── Summary ────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  Summary"
echo "================================================================"

if [[ $WANT_LAW -eq 1 ]]; then
    c=$(find "$DATA_ROOT/chunked"  -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
    e=$(find "$DATA_ROOT/embedded" -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1 || echo "?")
    printf "  case-law     chunked=%-6s embedded=%-6s size=%-6s %s\n" "$c" "$e" "$sz" "$DATA_ROOT"
fi

if [[ $WANT_LB -eq 1 ]] && [[ -f "$LEGALBENCH_DIR/manifest.json" ]]; then
    lc=$(python3 -c "import json;print(json.load(open('$LEGALBENCH_DIR/manifest.json'))['chunks'])" 2>/dev/null || echo "?")
    lq=$(python3 -c "import json;print(json.load(open('$LEGALBENCH_DIR/manifest.json'))['queries'])" 2>/dev/null || echo "?")
    sz=$(du -sh "$LEGALBENCH_DIR" 2>/dev/null | cut -f1 || echo "?")
    printf "  legalbench   chunks=%-6s queries=%-6s size=%-6s %s\n" "$lc" "$lq" "$sz" "$LEGALBENCH_DIR"
fi

echo
if [[ $FAILED -eq 0 ]]; then
    echo "ok"
    exit 0
else
    echo "one or more components failed; see messages above" >&2
    exit 1
fi
