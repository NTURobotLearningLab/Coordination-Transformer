#!/bin/bash
# Recreate the CooT conda env.
# Versions recovered from the last working env (wandb run 2026-04-30): Python 3.9.15, torch 1.13.0+cu117.
# Usage: bash setup_env.sh [env_name]
set -e

ENV_NAME=${1:-coot}
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Python
conda create -y -n "$ENV_NAME" python=3.9.15
conda activate "$ENV_NAME"
python -m pip install --upgrade "pip<25.1" "setuptools==75.8.0" wheel

# 2. PyTorch (cu117 wheels bundle their own CUDA runtime; only needs a recent NVIDIA driver)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. Python deps
pip install -r "$REPO_DIR/requirements.txt"

# 4. zsceval (Overcooked envs + HSP/MEP partner policies), editable
pip install -e "$REPO_DIR/zsceval"

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo "Done. Activate with: conda activate $ENV_NAME"
