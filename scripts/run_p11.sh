#!/bin/bash
# Submit P11: one CPU job per fitted spec (all scenarios x seeds 0-2), then the collector.
cd /home/caroyf/myproject/ACMEC
ids=()
for spec in g0_fe g0_nofe g0.5_fe g0.5_nofe; do
  ids+=($(sbatch --parsable --job-name=p11_$spec --cpus-per-task=2 --mem=48G --time=6:00:00 slurm/cpu.sbatch scripts/p11_spec.sh $spec))
done
dep=$(IFS=:; echo "${ids[*]}")
sbatch --parsable --job-name=p11_collect --dependency=afterok:$dep --cpus-per-task=2 --mem=48G --time=1:00:00 \
  slurm/cpu.sbatch scripts/p11_generative.py --collect
echo "submitted: ${ids[*]}"
