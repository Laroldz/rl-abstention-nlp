#!/bin/bash
# One-time setup of the conda environment on TAMU Grace:
#   sbatch setup_env.slurm        (or: bash setup_env.slurm on a login node)
# Re-running is safe: it reuses the env if it exists and fixes its Python version if needed.

#SBATCH --job-name=setup_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/group/nlp_abstain/rl-abstention-nlp
#SBATCH -o setup_env-%j.out
##SBATCH --account=<your_project_number>   # uncomment if you have more than one HPRC project

set -euo pipefail

ENV_PATH=/scratch/group/nlp_abstain/conda/rl-abstention
REPO_DIR=/scratch/group/nlp_abstain/rl-abstention-nlp
PY_VERSION=3.11

module purge
module load Anaconda3                      # check with: module spider Anaconda3
module load WebProxy                       # compute nodes need this to reach PyPI

# keep conda's package cache off the small home quota
export CONDA_PKGS_DIRS=/scratch/group/nlp_abstain/conda/pkgs
export PIP_CACHE_DIR=/scratch/group/nlp_abstain/conda/pip-cache

if [ ! -d "$ENV_PATH" ]; then
    echo ">> Creating env at $ENV_PATH (Python $PY_VERSION)"
    conda create -p "$ENV_PATH" python=$PY_VERSION -y
fi

set +u   # conda's activate scripts reference unset variables
source activate "$ENV_PATH"
export PATH="$ENV_PATH/bin:$PATH"   # make sure the env python wins inside batch jobs
export CONDA_PREFIX="$ENV_PATH"
set -u
hash -r

if ! python --version 2>&1 | grep -q "Python $PY_VERSION"; then
    echo ">> Env has $(python --version 2>&1), switching to Python $PY_VERSION"
    conda install -p "$ENV_PATH" python=$PY_VERSION -y
    hash -r
fi
echo ">> Using $(which python): $(python --version 2>&1)"

echo ">> Installing requirements"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"

echo ">> Checking install"
python - <<'PY'
import torch, transformers, trl, peft, vllm, datasets
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)
print("vllm", vllm.__version__, "| datasets", datasets.__version__)
PY

mkdir -p "$REPO_DIR/logs"
echo ">> Done. Activate later with:  module load Anaconda3 && source activate $ENV_PATH"
