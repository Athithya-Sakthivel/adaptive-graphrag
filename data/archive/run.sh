
cd /workspace/data/archive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -q

sudo apt-get install -y lbzip2  


export HF_TOKEN=$HF_TOKEN # https://huggingface.co/settings/tokens/new?tokenType=write
export HF_USER_NAME=Athithya117
export DATASET_SIZE=medium

python3 download_datasets.py --dir /workspace/tmp/data/raw --scale $DATASET_SIZE


# python3 upload_datasets.py