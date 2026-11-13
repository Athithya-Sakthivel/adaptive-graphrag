cd /workspace/src/utils/ && python3 -m venv .venv
source /workspace/src/utils/.venv/bin/activate
pip install -r requirements.inspect.txt -q


cd /workspace
python3 src/utils/inspect_chunks.py \
  --chunked-dir /workspace/tmp/data/chunked \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16

python3 src/utils/inspect_graph.py

python3 src/utils/inspect_data.py /workspace/tmp/data/raw \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders \
  --sample 500 --chunker-sample 30

python3 src/utils/inspect_data.py /workspace/tmp/data/raw \
  --tokenizer-path /workspace/src/indexing_pipeline/embedders \
  --inspect-one 1088107
cd /workspace/src
