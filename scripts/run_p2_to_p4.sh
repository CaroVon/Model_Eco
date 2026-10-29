#!/bin/bash
# Rebuild edges/nodes (with placeholder parents), descriptives, and the 10% choice sample; then full P4 runs.
# usage: bash scripts/run_p2_to_p4.sh   (submits Slurm jobs)
cd /home/caroyf/myproject/ACMEC
a=$(sbatch --parsable --job-name=p2p3 slurm/cpu.sbatch scripts/run_p2_p3_frac.sh)
for spec in main_strat main_unif; do
  sbatch --parsable --job-name=p4_$spec --dependency=afterok:$a slurm/cpu.sbatch scripts/p4_replay.py --spec $spec
done
