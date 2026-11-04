# Source this on the pod before running anything:  source scripts/env.sh
# Weights and the venv live on the container disk (allow at least 250 GB); /workspace is a network
# filesystem (fine for code and results, slow for multi-GB weights).
export HF_HOME=/root/hf
# The pod is capped at 16 vCPUs even though nproc reports the host's 128 cores.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
if [ -f /root/venv/bin/activate ]; then source /root/venv/bin/activate; fi
