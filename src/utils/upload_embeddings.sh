#!/usr/bin/env bash
# Upload the pipeline's embedded/ directory to a private Hugging Face dataset.
#
# Required env:
#   HF_USER_NAME   your HF username, e.g. Athithya117
#
# Optional env:
#   HF_TOKEN       WRITE token. If unset, falls back to a saved `hf auth login`.
#                  Create at https://huggingface.co/settings/tokens/new?tokenType=write
#   HF_DATASET_NAME  default: legalrag-embeddings
#   EMBEDDED_DIR     default: /workspace/tmp/data/embedded
#
# Usage:
#   export HF_USER_NAME=Athithya117
#   export HF_TOKEN=hf_...
#   ./src/utils/upload_embeddings.sh [EMBEDDED_DIR]
#
# The dataset is created if missing. Uploads are resumable — rerun after a
# network drop and only changed files are re-sent.

set -euo pipefail

EMBEDDED_DIR="${1:-${EMBEDDED_DIR:-/workspace/tmp/data/embedded}}"
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

[[ -d "$EMBEDDED_DIR" ]] || {
    echo "error: embedded directory not found: $EMBEDDED_DIR" >&2
    exit 1
}

MANIFEST="$EMBEDDED_DIR/_manifest.jsonl"
[[ -f "$MANIFEST" ]] || {
    echo "error: _manifest.jsonl missing in $EMBEDDED_DIR" >&2
    echo "  Without it, load will refuse to run on the restored corpus." >&2
    exit 1
}

CASE_FILES=$(find "$EMBEDDED_DIR" -maxdepth 1 -type f -name '*.jsonl' ! -name '_*' | wc -l | tr -d ' ')
if [[ "$CASE_FILES" -eq 0 ]]; then
    echo "error: no case jsonl files in $EMBEDDED_DIR" >&2
    exit 1
fi

# ── Auth ─────────────────────────────────────────────────────────────────
# hf reads HF_TOKEN from the environment; it takes precedence over any saved
# login. A read token will fail at upload with 401 — you need a write token.

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
        echo "  export HF_TOKEN=hf_...   (must be a WRITE token for upload)" >&2
        exit 1
    fi
    echo "auth:    saved login ($SAVED_TOKEN)"
fi

# ── Summary ──────────────────────────────────────────────────────────────

echo "repo:    $REPO_ID"
echo "type:    $REPO_TYPE (private)"
echo "source:  $EMBEDDED_DIR"
echo "files:   $CASE_FILES case jsonl + _manifest.jsonl"
echo "size:    $(du -sh "$EMBEDDED_DIR" | cut -f1)"
echo

# ── Create repo if missing ───────────────────────────────────────────────

if ! hf repo create "$REPO_ID" --repo-type "$REPO_TYPE" --private --exist-ok >/dev/null 2>&1; then
    # Older CLI builds lack --exist-ok. Tolerate "already exists".
    if ! hf repo create "$REPO_ID" --repo-type "$REPO_TYPE" --private >/dev/null 2>&1; then
        echo "note: repo $REPO_ID already exists (or could not be created; upload will try anyway)"
    fi
fi

# ── Upload ───────────────────────────────────────────────────────────────
# upload-large-folder is the resumable path for many-files directories.
# It uploads the directory CONTENTS to the repo root, so a later
# `hf download ... --local-dir <dest>` restores exactly this shape.
# `.cache` is excluded in case a previous download left one behind.

echo "uploading (resumable — safe to Ctrl-C and rerun)..."
hf upload-large-folder "$REPO_ID" "$EMBEDDED_DIR" \
    --repo-type "$REPO_TYPE" \
    --exclude ".cache/**" \
    --exclude "**/.cache/**"

echo
echo "upload complete: https://huggingface.co/datasets/$REPO_ID"
echo
echo "restore on laptop:"
echo "  HF_USER_NAME=$HF_USER_NAME ./src/utils/download_embeddings.sh /workspace/tmp/data/embedded"
