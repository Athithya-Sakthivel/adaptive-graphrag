#!/usr/bin/env bash
# Restore the pipeline's embedded/ directory from a private HF dataset.
#
# Required env:
#   HF_USER_NAME   your HF username, e.g. Athithya117
#
# Optional env:
#   HF_TOKEN       READ token. If unset, falls back to a saved `hf auth login`.
#                  Create at https://huggingface.co/settings/tokens/new?tokenType=read
#   HF_DATASET_NAME  default: legalrag-embeddings
#   EMBEDDED_DIR     default: /workspace/tmp/data/embedded
#
# Usage:
#   export HF_USER_NAME=Athithya117
#   export HF_TOKEN=hf_...
#   ./src/utils/download_embeddings.sh [/workspace/tmp/data/embedded]
#
# Resumable: unchanged files are skipped by hash. Safe to rerun.

set -euo pipefail

DEST="${1:-${EMBEDDED_DIR:-/workspace/tmp/data/embedded}}"
HF_USER_NAME="${HF_USER_NAME:?HF_USER_NAME is required. export HF_USER_NAME=Athithya117}"
HF_DATASET_NAME="${HF_DATASET_NAME:-legalrag-embeddings}"
REPO_ID="${HF_USER_NAME}/${HF_DATASET_NAME}"
REPO_TYPE="dataset"

# ── Preflight ────────────────────────────────────────────────────────────

command -v hf >/dev/null 2>&1 || {
    echo "error: hf CLI not found." >&2
    echo "  pip install 'huggingface_hub>=0.30'" >&2
    exit 1
}

# ── Auth ─────────────────────────────────────────────────────────────────

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    echo "auth:    HF_TOKEN from environment"
else
    SAVED_TOKEN=""
    for p in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        [[ -f "$p" ]] && SAVED_TOKEN="$p" && break
    done
    if [[ -z "$SAVED_TOKEN" ]]; then
        echo "error: no HF_TOKEN and no saved hf login found" >&2
        echo "  export HF_TOKEN=hf_...   (a READ token is sufficient for download)" >&2
        exit 1
    fi
    echo "auth:    saved login ($SAVED_TOKEN)"
fi

# ── Summary ──────────────────────────────────────────────────────────────

mkdir -p "$DEST"

echo "repo:    $REPO_ID"
echo "target:  $DEST"
echo

# ── Download ─────────────────────────────────────────────────────────────
# hf download creates DEST/.cache/huggingface for its own metadata. That is
# harmless: the pipeline's _cleanup_orphans only iterates *.jsonl at DEST
# top-level, and read_jsonl only opens files you name explicitly.

echo "downloading (resumable)..."
hf download "$REPO_ID" --repo-type "$REPO_TYPE" --local-dir "$DEST"

# ── Verify ───────────────────────────────────────────────────────────────

CASE_FILES=$(find "$DEST" -maxdepth 1 -type f -name '*.jsonl' ! -name '_*' | wc -l | tr -d ' ')
echo
echo "restored: $CASE_FILES case jsonl files"

if [[ ! -f "$DEST/_manifest.jsonl" ]]; then
    echo "error: _manifest.jsonl not present after download" >&2
    echo "  load will refuse to run without it." >&2
    exit 1
fi

echo "manifest: $DEST/_manifest.jsonl"
echo
echo "next:"
echo "  cd /workspace/src"
echo "  python3 -m indexing_pipeline.cli load"
