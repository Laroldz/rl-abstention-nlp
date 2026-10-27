#!/bin/bash

#SBATCH --job-name=baseline_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/scratch/group/nlp_abstain/rl-abstention-nlp   # repo on Grace; logs, CSVs and outputs resolve here
#SBATCH -o logs/%x-%j.out
##SBATCH --account=<your_project_number>   # uncomment if you have more than one HPRC project

module purge
module load Anaconda3                      # check with: module spider Anaconda3

# use the env's python directly; "source activate" doesn't work reliably inside batch jobs
ENV_PATH=/scratch/group/nlp_abstain/conda/rl-abstention   # created by setup_env.slurm
export PATH=$ENV_PATH/bin:$PATH
export CONDA_PREFIX=$ENV_PATH
hash -r
python -c "import datasets, torch, trl, vllm" || { echo "conda env missing or incomplete: run  sbatch setup_env.slurm  first"; exit 1; }
echo "Using $(which python) ($(python --version 2>&1))"

# shared model cache for the group (home quota is small, and $SCRATCH differs per user)
export HF_HOME=/scratch/group/nlp_abstain/hf_cache
# no internet on Grace: model must already be in HF_HOME, MedMCQA comes from medmcqa_*.csv
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

nvidia-smi || true
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -u eval.py
