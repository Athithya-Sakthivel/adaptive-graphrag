
export HF_TOKEN=${HF_TOKEN}
export PG_PASS=${PG_PASS:-temprorypassword13r2}

export HF_LB_REVISION="513b8d9857fe13dde2af2c73f90dc0a0b063405b"
export HF_LAW_REVISION="690800ee6c49f0bfde6c1e26f64031601feba6ae" # 10000 corpus rows


docker rm -f legal-pgvector
docker rm -f legal-pgvector 2>/dev/null || true
rm -rf tmp/data/

bash src/utils/bootstrap.sh

bash src/utils/download_artifacts.sh


cd /workspace/src
source .venv/bin/activate
export DATABASE_URL="postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag"
export MODEL=/workspace/src/indexing_pipeline/embedders
export CORPUS_ROWS=10000

# Correct order: manifest → load → validate → index
python -m indexing_pipeline.cli drop-index --log-level INFO
python -m indexing_pipeline.cli manifest   --log-level INFO
python -m indexing_pipeline.cli load --model-dir "$MODEL" --gpu-model-dir "$MODEL" --log-level INFO

python -m indexing_pipeline.cli validate --model-dir "$MODEL" --gpu-model-dir "$MODEL" --log-level INFO 2>&1 | tee /tmp/validate.log

python -m indexing_pipeline.cli index --model-dir "$MODEL" --gpu-model-dir "$MODEL" --log-level INFO
