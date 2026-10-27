# Rewarding Intellectual Humility: Learning When Not to Answer in Large Language Models

## Datasets used:
MEDMCQA: https://huggingface.co/datasets/openlifescienceai/medmcqa

GSMK8: https://huggingface.co/datasets/openai/gsm8k

Hendrycks Math: https://huggingface.co/datasets/EleutherAI/hendrycks_math


## Code architecture
train.py: Used for Training RL or SFT by loading model

eval.py: Used for evaluation of model (base model or SFT-tuned model or RL-model)

rewards.py: Used for RL with idk reward weight, reward systems defined - format reward and accuracy reward

data.py: Used for loading the datasets

utils.py: Utility functions for extracting answers or resolving checkpoints


## Steps to execute:
### environment (TAMU Grace, run once)
```bash
sbatch setup_env.slurm   # creates /scratch/group/nlp_abstain/conda/rl-abstention (Python 3.11) and installs requirements
```
### IDK reward sweep (Qwen, MedMCQA, 5 RL runs + eval)
```bash
sbatch grace_rl_sweep.slurm
```
### training
```bash
# Set appropriate settings in train.py
# Modify job.sh as per the availability and include "python -u train.py" line at the bottom
sbatch job.sh
```
### evaluation
```bash
# Set appropriate settings in eval.py
# Modify job.sh as per the availability and include "python -u eval.py" line at the bottom
sbatch job.sh
```
