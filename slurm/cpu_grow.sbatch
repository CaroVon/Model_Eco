#!/bin/bash
#SBATCH --job-name=v2grow
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%A_%a.out
# usage: sbatch --array=0-24 slurm/cpu_grow.sbatch     (task id -> network x seed, see scripts/v2_grow.py)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa
source /home/caroyf/myproject/ACMEC/activate.sh
export OMP_NUM_THREADS=2
python scripts/v2_grow.py --task $SLURM_ARRAY_TASK_ID
