export HF_TOKEN=${HF_TOKEN}
export PG_PASS=${PG_PASS:-e6a1c2e5bf8d}

export HF_LB_REVISION="513b8d9857fe13dde2af2c73f90dc0a0b063405b"
export HF_LAW_REVISION="690800ee6c49f0bfde6c1e26f64031601feba6ae" # 10000 corpus rows


# docker volume rm legal_pgdata
# docker rm -f legal-pgvector
# docker rm -f legal-pgvector 2>/dev/null || true
# rm -rf tmp/data/

bash src/utils/bootstrap.sh


cd /workspace/src
source .venv/bin/activate
export DATABASE_URL="postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag"
export MODEL=/workspace/src/indexing_pipeline/embedders
export CORPUS_ROWS=10000

python -m indexing_pipeline.cli all \
  --embed-batch 32 \
  --inference-threads 8 \
  --log-level INFO
