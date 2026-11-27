#!/bin/bash
#SBATCH --job-name=v2mech
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%A_%a.out
# usage: UNTIL=2024-03 sbatch --array=0-63%16 --export=ALL slurm/cpu_mech.sbatch
#        (task id -> S5 subset id // 2, seed id % 2; network-level Shapley, scripts/v2_grow.py --s5-task)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa
source /home/caroyf/myproject/ACMEC/activate.sh
export OMP_NUM_THREADS=2
python scripts/v2_grow.py --s5-task $SLURM_ARRAY_TASK_ID ${UNTIL:+--until $UNTIL} ${NSEEDS:+--s5-nseeds $NSEEDS}
