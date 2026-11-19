#!/bin/bash
#SBATCH --job-name=acmec_gpup
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --qos=protected
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%j.out
# protected QOS (no preemption, <= 2h): usage: sbatch slurm/gpu_p.sbatch scripts/v2_main.py [args]
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa
source /home/caroyf/myproject/ACMEC/activate.sh
if [[ "$1" == *.sh ]]; then bash "$@"; else python "$@"; fi
