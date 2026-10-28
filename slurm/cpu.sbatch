#!/bin/bash
#SBATCH --job-name=acmec_cpu
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%j.out
# usage: sbatch slurm/cpu.sbatch scripts/p4_replay.py [args]
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa
source /home/caroyf/myproject/ACMEC/activate.sh
if [[ "$1" == *.sh ]]; then bash "$@"; else python "$@"; fi
