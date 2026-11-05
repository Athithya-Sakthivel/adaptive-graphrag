

cd /workspace/src/utils
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.inspect.txt -q
cd /workspace

# Full inspection
python3 src/utils/inspect_data.py /workspace/tmp/data/raw \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16 \
  --sample 500 --chunker-sample 30

# Single case deep-dive
python3 src/utils/inspect_data.py /workspace/tmp/data/raw \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16 \
  --inspect-one 1088107
cd /workspace/src




# 1. Persistent volume (survives container restarts)
docker volume create legal_pgdata
docker run -d \
  --name legal-pgvector \
  --restart unless-stopped \
  --shm-size=1g \
  -p 127.0.0.1:5432:5432 \
  -e POSTGRES_DB=legalrag \
  -e POSTGRES_USER=legalrag \
  -e POSTGRES_PASSWORD="$PG_PASS" \
  -v legal_pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:0.8.7-pg17-bookworm@sha256:7d0a41464797aeb9862dfdc0bbf1526b82cfd324cdd8036df19e3545f4a9ccb8 \
  -c shared_buffers=1GB \
  -c effective_cache_size=3GB \
  -c maintenance_work_mem=512MB \
  -c work_mem=16MB \
  -c temp_buffers=16MB \
  -c wal_buffers=32MB \
  -c max_wal_size=1GB \
  -c min_wal_size=256MB \
  -c checkpoint_timeout=15min \
  -c max_connections=20 \
  -c max_parallel_workers=2 \
  -c max_parallel_maintenance_workers=2


cd /workspace/src
rm -f /workspace/tmp/data/embedded/_HALTED
rm -f /workspace/tmp/data/embedded/_failures.jsonl
rm -f /workspace/tmp/data/embedded/_manifest.jsonl
rm -f /workspace/tmp/data/embedded/*.jsonl

sudo apt update
sudo apt install -y python3.14-dev build-essential




cd /workspace/src
source .venv/bin/activate
python -m pip install -q -r indexing_pipeline/requirements.txt pytest
python -m pytest tests/ -v

cd /workspace
find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
git add .
git commit -m "chunker v2.0: strip page markers, accept ':' terminal, sync gate_3+inspect_chunks, add terminal-sync test"

cd /workspace/src
source .venv/bin/activate
export CORPUS_ROWS=1
export DATABASE_URL="postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag"
python -m indexing_pipeline.cli all --log-level INFO



cd /workspace
source /workspace/src/utils/.venv/bin/activate
python3 src/utils/inspect_chunks.py \
  --chunked-dir /workspace/tmp/data/chunked \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16


cd /workspace
source /workspace/src/utils/.venv/bin/activate
python3 src/utils/inspect_graph.py


mkdir -p /workspace/src/indexing_pipeline/embedder

hf download Athithya117/dinghy-law-0.6b-v1-openvino-fp16 \
  --repo-type model \
  --local-dir /workspace/src/indexing_pipeline/embedder


find src \
  \( -type d \( -name '.venv' -o -name 'embedder' \) -prune \) -o \
  \( -type f -name '*.py' -print0 \) |
  xargs -0 -r pre-commit run basedpyright --files
