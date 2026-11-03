#!/usr/bin/env bash
# (Re)install the inference + analysis stack into /root/venv. Safe to re-run: uv reuses its cache.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
uv pip install --python /root/venv/bin/python vllm sentence-transformers pandas matplotlib
/root/venv/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
echo "SETUP DONE"
