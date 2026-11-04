#!/usr/bin/env bash
# Upload chunked/ and embedded/ to a single private HF dataset.
#
# Required env:
#   HF_USER_NAME   e.g. Athithya117
#   HF_TOKEN       write token (https://huggingface.co/settings/tokens/new?tokenType=write)
#
# Optional env:
#   HF_DATASET_NAME   default: legalrag-artifacts
#   UPLOAD_ANYWAY     set to 1 to upload despite _failures.jsonl being non-empty
#
# Usage:
#   HF_USER_NAME=Athithya117 HF_TOKEN=hf_... \
#     ./src/utils/upload_artifacts.sh [/kaggle/working/tmp/data]

set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-/workspace/tmp/data}}"
HF_USER_NAME="${HF_USER_NAME:?HF_USER_NAME is required}"
HF_DATASET_NAME="${HF_DATASET_NAME:-legalrag-artifacts}"
REPO_ID="${HF_USER_NAME}/${HF_DATASET_NAME}"

command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found (pip install 'huggingface_hub>=0.30')" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "error: $DATA_ROOT missing" >&2; exit 1; }

for sub in chunked embedded; do
    [[ -d "$DATA_ROOT/$sub" ]] || { echo "error: $DATA_ROOT/$sub missing" >&2; exit 1; }
    [[ -f "$DATA_ROOT/$sub/_manifest.jsonl" ]] || { echo "error: $DATA_ROOT/$sub/_manifest.jsonl missing" >&2; exit 1; }
    if [[ -f "$DATA_ROOT/$sub/_failures.jsonl" ]]; then
        N=$(wc -l < "$DATA_ROOT/$sub/_failures.jsonl")
        if [[ "$N" -gt 0 && "${UPLOAD_ANYWAY:-0}" != "1" ]]; then
            echo "error: $DATA_ROOT/$sub/_failures.jsonl has $N entries" >&2
            echo "  fix the failures and re-run, or set UPLOAD_ANYWAY=1 to override" >&2
            exit 1
        fi
    fi
done

if [[ -n "${HF_TOKEN:-}" ]]; then export HF_TOKEN; fi

CHUNKED_N=$(find "$DATA_ROOT/chunked"  -maxdepth 1 -name '*.jsonl' ! -name '_*' | wc -l | tr -d ' ')
EMBEDDED_N=$(find "$DATA_ROOT/embedded" -maxdepth 1 -name '*.jsonl' ! -name '_*' | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -shc "$DATA_ROOT/chunked" "$DATA_ROOT/embedded" | tail -1 | cut -f1)

echo "repo:     $REPO_ID (private)"
echo "chunked:  $CHUNKED_N cases"
echo "embedded: $EMBEDDED_N cases"
echo "size:     $TOTAL_SIZE"
echo

hf repo create "$REPO_ID" --repo-type dataset --private --exist-ok >/dev/null 2>&1 || true

hf upload-large-folder "$REPO_ID" "$DATA_ROOT" \
    --repo-type dataset \
    --exclude "raw/**" \
    --exclude "manifest/**" \
    --exclude "reports/**" \
    --exclude "*.lock" \
    --exclude ".cache/**" \
    --exclude "**/.cache/**"

echo
echo "uploaded: https://huggingface.co/datasets/$REPO_ID"
