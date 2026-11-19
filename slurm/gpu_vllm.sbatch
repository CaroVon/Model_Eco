#!/bin/bash
#SBATCH --job-name=acmec_vllm
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%j.out
# GPU job in the conda env qc-vllm, BORROWED read-only from the ICML project (user approval 2026-10-31; see README
# "Borrowed resources"). Do not pip-install into this env. usage: sbatch slurm/gpu_vllm.sbatch scripts/p22_embed.py --embed
echo "[borrowed] conda env qc-vllm (ICML project), read-only; $(date)"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate qc-vllm
source /home/caroyf/myproject/ACMEC/activate.sh
python "$@"
