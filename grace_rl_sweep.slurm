#!/bin/bash

# RL (GRPO) sweep over the IDK abstention reward: Qwen3-4B on MedMCQA, one array task per reward.
# Each task trains, then evaluates its final model on the test split.
#
#   sbatch grace_rl_sweep.slurm                                   # full sweep
#   sbatch --array=1 --time=01:00:00 --export=ALL,MAX_STEPS=2,EVAL_SAMPLES=20 grace_rl_sweep.slurm   # smoke test
#
# Resubmitting after a timeout resumes each run from its last checkpoint.
# Untrained Qwen reference run (no RL):  MODEL=qwen DATA=medmcqa NUM_SAMPLES=5000 python -u eval.py

#SBATCH --job-name=rl_idk_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --chdir=/scratch/group/nlp_abstain/rl-abstention-nlp   # repo on Grace; logs, CSVs and outputs resolve here
#SBATCH -o logs/%x-%A_%a.out
##SBATCH --account=<your_project_number>   # uncomment if you have more than one HPRC project

IDK_REWARDS=(-0.5 0 0.25 0.5 0.75)
# one-off run with any value:  sbatch --array=0 --export=ALL,IDK_OVERRIDE=-0.25 grace_rl_sweep.slurm
IDK=${IDK_OVERRIDE:-${IDK_REWARDS[$SLURM_ARRAY_TASK_ID]}}

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # vLLM sleep mode doesn't support expandable segments

# Qwen-4B OOMs on a 40GB A100 with 8 completions per pass; 2 x 256 keeps the same 512 completions per update
export PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-256}

# shared by train.py, rewards.py and eval.py
export DATA=medmcqa
export IDK_REWARD=$IDK
export OUTPUT_DIR=rl_medmcqa_qwen_idk$IDK

echo "Task $SLURM_ARRAY_TASK_ID: IDK reward = $IDK, output = $OUTPUT_DIR"
nvidia-smi || true

# Skip training if this run already finished; resubmitting then only runs eval.
# Success is judged by the saved model, not the exit code: vLLM often segfaults at shutdown after saving.
if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    TRAINING_TYPE=rl BASE_MODEL=qwen RESUME_FROM_CHECKPOINT=true \
        python -u train.py
fi
[ -f "$OUTPUT_DIR/final_model/adapter_config.json" ] || { echo "Training did not finish ($OUTPUT_DIR/final_model missing)"; exit 1; }

LOAD_SPECIFIC_MODEL=true MODEL=qwen MODEL_CHECKPOINT_PATH=$OUTPUT_DIR/final_model \
    EVAL_TYPE=rl RUN_TAG=idk$IDK NUM_SAMPLES=${EVAL_SAMPLES:-5000} \
    python -u eval.py
