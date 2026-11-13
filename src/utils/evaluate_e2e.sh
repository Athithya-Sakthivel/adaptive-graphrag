export HF_TOKEN=${HF_TOKEN}
export PG_PASS=${PG_PASS:-e6a1c2e5bf8d}



# docker rm -f legal-pgvector && docker volume rm legal_pgdata
# rm -rf tmp/data/

bash src/utils/bootstrap.sh



cd /workspace/src
source .venv/bin/activate
export PG_PASS=${PG_PASS:-e6a1c2e5bf8d}
export DATABASE_URL="postgresql://legalrag:${PG_PASS}@127.0.0.1:5432/legalrag"
export MODEL=/workspace/src/indexing_pipeline/embedders
export CORPUS_ROWS=50000

python -m indexing_pipeline.cli drop-index # to index faster
python -m indexing_pipeline.cli all --embed-batch 32 --inference-threads 8 --log-level INFO




# python -m indexing_pipeline.cli drop-index && python -m indexing_pipeline.cli load --log-level INFO && python -m indexing_pipeline.cli validate --log-level INFO && python -m indexing_pipeline.cli index --log-level INFO

bash indexing_pipeline/preprocess.sh

# mv /workspace/tmp/data/chunked/_manifest.jsonl.full /workspace/tmp/data/chunked/_manifest.jsonl && python -m indexing_pipeline.cli embed --log-level INFO

# Once finished, load the new 90K cases into the DB
python -m indexing_pipeline.cli load --log-level INFO
