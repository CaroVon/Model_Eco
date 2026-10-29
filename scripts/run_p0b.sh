#!/bin/bash
# Submit P0b: every scenario x seed as one GPU job, then the collector after all succeed.
# usage: bash scripts/run_p0b.sh [seeds...]   (default seeds 42 43 44)
cd /home/caroyf/myproject/ACMEC
seeds=${@:-42 43 44}
ids=()
for sc in pa fit mix fit_decay mix_decay fit_bulk mix_bulk; do
  for s in $seeds; do
    ids+=($(sbatch --parsable --job-name=p0b_${sc}_${s} --cpus-per-task=4 --mem=48G --time=12:00:00 \
      slurm/gpu.sbatch scripts/p0b_identification.py --scenario $sc --seed $s))
  done
done
dep=$(IFS=:; echo "${ids[*]}")
sbatch --parsable --job-name=p0b_collect --dependency=afterok:$dep --cpus-per-task=2 --mem=8G --time=0:30:00 \
  slurm/cpu.sbatch scripts/p0b_collect.py
echo "submitted ${#ids[@]} jobs: ${ids[*]}"
