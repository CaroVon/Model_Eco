#!/bin/bash
#SBATCH --job-name=nbexec
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/%x_%j.out
# usage: sbatch slurm/nb.sbatch notebooks/note3_method.ipynb
source ~/miniconda3/etc/profile.d/conda.sh; conda activate hfpa
source /home/caroyf/myproject/ACMEC/activate.sh
cd /home/caroyf/myproject/ACMEC/notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.startup_timeout=900 "$(basename "$1")"
