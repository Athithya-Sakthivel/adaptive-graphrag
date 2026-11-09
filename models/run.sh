

cd /workspace/models
python3 -m venv .venv
source .venv/bin/activate


python3 -m pip install --upgrade pip setuptools wheel -q

pip install -r requirements.txt -q

pip uninstall -y torchvision


export OMP_NUM_THREADS=8
export BATCH_SIZE=2
export OV_PERFORMANCE_HINT=LATENCY

python -u convert_and_test.py 2>&1 | tee openvino_export.log
