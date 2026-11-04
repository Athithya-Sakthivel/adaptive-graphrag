#!/usr/bin/env bash
# Restore chunked/ and embedded/ from the private HF dataset.
#
# Required env:
#   HF_USER_NAME   e.g. Athithya117
#   HF_TOKEN       read token (write also works)
#
# Optional env:
#   HF_DATASET_NAME   default: legalrag-artifacts
#
# Usage:
#   HF_USER_NAME=Athithya117 HF_TOKEN=hf_... \
#     ./src/utils/download_artifacts.sh [/workspace/tmp/data]

set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-/workspace/tmp/data}}"
HF_USER_NAME="${HF_USER_NAME:?HF_USER_NAME is required}"
HF_DATASET_NAME="${HF_DATASET_NAME:-legalrag-artifacts}"
REPO_ID="${HF_USER_NAME}/${HF_DATASET_NAME}"

command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found" >&2; exit 1; }

if [[ -n "${HF_TOKEN:-}" ]]; then export HF_TOKEN; fi

mkdir -p "$DATA_ROOT"

echo "repo:   $REPO_ID"
echo "target: $DATA_ROOT"
echo

hf download "$REPO_ID" --repo-type dataset --local-dir "$DATA_ROOT"

CHUNKED_N=$(find "$DATA_ROOT/chunked"  -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
EMBEDDED_N=$(find "$DATA_ROOT/embedded" -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')

echo
echo "restored chunked=$CHUNKED_N  embedded=$EMBEDDED_N"

for sub in chunked embedded; do
    [[ -f "$DATA_ROOT/$sub/_manifest.jsonl" ]] || {
        echo "error: $DATA_ROOT/$sub/_manifest.jsonl missing after download" >&2
        exit 1
    }
done

echo "ok"
