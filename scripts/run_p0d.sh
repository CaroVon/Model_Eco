#!/bin/bash
# Submit P0d: every scenario x seed as one GPU job, then the collector after all succeed.
# usage: bash scripts/run_p0c.sh [seeds...]   (default seeds 42-46)
cd /home/caroyf/myproject/ACMEC
seeds=${@:-42 43 44 45 46}
ids=()
for sc in pa fit mix fit_decay mix_decay fit_bulk mix_bulk; do
  for s in $seeds; do
    ids+=($(sbatch --parsable --job-name=p0d_${sc}_${s} --cpus-per-task=4 --mem=96G --time=12:00:00 \
      slurm/gpu.sbatch scripts/p0d_gmm.py --scenario $sc --seed $s))
  done
done
dep=$(IFS=:; echo "${ids[*]}")
sbatch --parsable --job-name=p0d_collect --dependency=afterok:$dep --cpus-per-task=2 --mem=8G --time=0:30:00 \
  slurm/cpu.sbatch scripts/p0d_collect.py
echo "submitted ${#ids[@]} jobs: ${ids[*]}"
