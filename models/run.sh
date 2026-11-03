

cd /workspace/models
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.onnx.txt


python3 convert_and_test.py


# 1. Download the correct Ubuntu x64 CPU asset (tar.gz, not zip)
wget https://github.com/ggml-org/llama.cpp/releases/download/b12058/llama-b12058-bin-ubuntu-x64.tar.gz

# 2. Extract into a dedicated directory
mkdir -p llama-cpp-b12058
tar -xzf llama-b12058-bin-ubuntu-x64.tar.gz -C llama-cpp-b12058

# 3. Find and verify the server binary
cd llama-cpp-b12058
find . -name "llama-server" -type f
# The binary is typically at ./build/bin/llama-server or ./bin/llama-server


cd /workspace/data/
source .venv/bin/activate


hf download Athithya117/LFM2.5-1.2B-Instruct-GGUF \
  LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
  --local-dir /workspace/models/llm

hf download Athithya117/dinghy-law-0.6b-v1-gguf \
  dinghy-law-0.6b-v1-Q8_0.gguf \
  --local-dir /workspace/models/embedder
