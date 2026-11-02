cd /workspace/data/

python3 -m venv .venv
pip install huggingface_hub==2.4.0


export HF_TOKEN=$HF_TOKEN # https://huggingface.co/settings/tokens/new?tokenType=read

export DATASET_SIZE=min


python3 download_raw_data.py /workspace/tmp/data/raw