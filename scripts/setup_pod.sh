#!/usr/bin/env bash
# One-time setup on the Runpod pod (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# Run from the repo root:  bash scripts/setup_pod.sh   (log: logs/setup.log)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
source scripts/env.sh

# 1. venv on the container disk, with just enough to start downloading weights.
uv venv /root/venv --python 3.12 --seed
uv pip install --python /root/venv/bin/python huggingface_hub hf_xet

# 2. Download all weights in the background while vLLM installs.
nohup /root/venv/bin/python scripts/download_models.py > logs/download.log 2>&1 &
echo "download started (pid $!), see logs/download.log"

# 3. Inference + analysis stack.
uv pip install --python /root/venv/bin/python vllm==0.33.0 transformers==5.20.2 sentence-transformers==6.2.0 scikit-learn==1.9.1 pandas==3.1.0 matplotlib==3.11.2
/root/venv/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
echo "SETUP DONE"
