#!/usr/bin/env bash
# Restore chunked/ and embedded/ from the private HF dataset.
#
# Primary path:  download tarballs (2 requests) and extract them.
# Fallback path: per-file download (3000+ requests) — used only if tarballs
#                are absent, e.g. against a dataset uploaded by an older
#                Block 2. Expect HF rate-limit backoffs on that path; it
#                will eventually complete but slowly.
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

# ── Primary path: tarballs ─────────────────────────────────────────
echo "attempting tarball download (2 requests)..."
TARBALL_OK=0

if hf download "$REPO_ID" \
        --repo-type dataset \
        --include "chunked.tar.gz" \
        --include "embedded.tar.gz" \
        --local-dir "$DATA_ROOT" 2>/dev/null; then
    if [[ -f "$DATA_ROOT/chunked.tar.gz" && -f "$DATA_ROOT/embedded.tar.gz" ]]; then
        TARBALL_OK=1
    fi
fi

if [[ "$TARBALL_OK" -eq 1 ]]; then
    echo "extracting tarballs..."
    rm -rf "$DATA_ROOT/chunked" "$DATA_ROOT/embedded"
    tar -xzf "$DATA_ROOT/chunked.tar.gz"  -C "$DATA_ROOT"
    tar -xzf "$DATA_ROOT/embedded.tar.gz" -C "$DATA_ROOT"
    rm -f "$DATA_ROOT/chunked.tar.gz" "$DATA_ROOT/embedded.tar.gz"
    echo "extracted from tarballs"
else
    # ── Fallback path: per-file download ──────────────────────────
    echo "no tarballs found — falling back to per-file download"
    echo "  (this will hit HF rate limits for >1000 files; backoffs expected)"
    echo
    hf download "$REPO_ID" --repo-type dataset --local-dir "$DATA_ROOT"
fi

# ── Counts ─────────────────────────────────────────────────────────
CHUNKED_N=$(find "$DATA_ROOT/chunked"  -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')
EMBEDDED_N=$(find "$DATA_ROOT/embedded" -maxdepth 1 -name '*.jsonl' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')

echo
echo "restored chunked=$CHUNKED_N  embedded=$EMBEDDED_N"

# ── Manifest presence ──────────────────────────────────────────────
for sub in chunked embedded; do
    [[ -f "$DATA_ROOT/$sub/_manifest.jsonl" ]] || {
        echo "error: $DATA_ROOT/$sub/_manifest.jsonl missing after download" >&2
        exit 1
    }
done

# ── Manifest-vs-disk verification ─────────────────────────────────
python3 - <<PYEOF
import json, sys
from pathlib import Path

data_root = Path("$DATA_ROOT")
exit_code = 0

for sub in ("chunked", "embedded"):
    manifest_path = data_root / sub / "_manifest.jsonl"
    if not manifest_path.is_file():
        sys.exit(f"error: {manifest_path} missing")

    expected = {
        json.loads(line)["case_id"]
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    }
    actual = {
        int(p.stem)
        for p in (data_root / sub).glob("*.jsonl")
        if p.stem.isdigit()
    }

    missing = expected - actual
    extra = actual - expected

    if missing:
        print(f"  {sub}: MISSING {len(missing)} files listed in manifest", file=sys.stderr)
        for m in sorted(missing)[:10]:
            print(f"    {m}.jsonl", file=sys.stderr)
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more", file=sys.stderr)
        exit_code = 1

    if extra:
        print(f"  {sub}: {len(extra)} stale files on disk not in manifest "
              f"(will be auto-cleaned by chunk stage)")

if exit_code == 0:
    print("verification: all manifest entries present on disk")
sys.exit(exit_code)
PYEOF

echo "ok"
