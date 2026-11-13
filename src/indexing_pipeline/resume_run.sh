cd /workspace/src
source .venv/bin/activate
export PG_PASS=${PG_PASS:-e6a1c2e5bf8d}
export DATABASE_URL="postgresql://legalrag:${PG_PASS}@127.0.0.1:5432/legalrag"
export MODEL=/workspace/src/indexing_pipeline/embedders
export CORPUS_ROWS=100000

# 1. Restore the full 100K manifest
mv /workspace/tmp/data/chunked/_manifest.jsonl.full /workspace/tmp/data/chunked/_manifest.jsonl || true

# 2. Resume embedding (will skip the 3,665 already done and process the remaining ~96K)
python -m indexing_pipeline.cli embed --log-level INFO

# 3. Drop the current 3.6K indexes to prepare for the full dataset load
python -m indexing_pipeline.cli drop-index

# 4. Load the new chunks into the database
python -m indexing_pipeline.cli load --log-level INFO

# 5. Validate the combined dataset
python -m indexing_pipeline.cli validate --log-level INFO

# 6. Rebuild the indexes on the full 100K dataset
python -m indexing_pipeline.cli index --log-level INFO
