
hf download Athithya117/bge-small-en-v1.5-openvino-qint8 \
  --repo-type model \
  --local-dir /workspace/src/indexing_pipeline/embedders


docker run -d \
  --name legal-pgvector \
  --restart unless-stopped \
  --shm-size=2g \
  -p 127.0.0.1:5432:5432 \
  -e POSTGRES_DB=legalrag \
  -e POSTGRES_USER=legalrag \
  -e POSTGRES_PASSWORD="$PG_PASS" \
  -v legal_pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:0.8.7-pg17-bookworm@sha256:7d0a41464797aeb9862dfdc0bbf1526b82cfd324cdd8036df19e3545f4a9ccb8 \
  -c shared_buffers=2GB \
  -c effective_cache_size=4GB \
  -c maintenance_work_mem=1GB \
  -c work_mem=16MB \
  -c temp_buffers=16MB \
  -c wal_buffers=64MB \
  -c max_wal_size=2GB \
  -c min_wal_size=512MB \
  -c checkpoint_timeout=15min \
  -c max_connections=20 \
  -c max_parallel_workers=2 \
  -c max_parallel_maintenance_workers=2 \
  -c random_page_cost=1.1 \
  -c effective_io_concurrency=200


cd /workspace/src && python3 -m venv .venv && source .venv/bin/activate
pip install -r indexing_pipeline/requirements.txt

python3 ../data/download_raw_data.py --dir /workspace/tmp/data/raw \
    --scale medium \
    --force
