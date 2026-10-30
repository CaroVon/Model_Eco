#!/bin/bash
# P15-P16 after `python scripts/p15_decomp.py --part build`. usage: bash scripts/run_p15.sh
cd /home/caroyf/myproject/ACMEC
a=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_main.sh)
b=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_decomp.py --part shapley)
c=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_mix.sh)
echo "main $a  shapley $b  mixture $c"
