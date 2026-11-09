#!/usr/bin/env bash
# Populate ./embedder with the FastEmbed model cache. Idempotent.
#
# FastEmbed maps friendly names (BAAI/bge-small-en-v1.5) to ONNX repos
# (Qdrant/bge-small-en-v1.5-onnx-Q). The resulting directory name will
# reflect the *underlying* repo, not the friendly name — that's expected
# and is handled at load time by host_dense._resolve_model_path().
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${DENSE_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
mkdir -p embedder

echo "[prepare] downloading '${MODEL}' into ./embedder"
python3 - <<PY
from fastembed import TextEmbedding
TextEmbedding(model_name="${MODEL}", cache_dir="./embedder")
PY

echo "[prepare] contents of embedder/:"
for d in embedder/models--*; do
  [ -d "$d" ] || continue
  echo "  $d"
  for s in "$d"/snapshots/*/; do
    [ -d "$s" ] || continue
    echo "    snapshot: $s"
    ls -1 "$s" | sed 's/^/      /'
  done
done

echo "[prepare] done."
