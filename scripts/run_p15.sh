#!/bin/bash
# P15-P17 after `python scripts/p15_decomp.py --part build` (and P18 text features). usage: bash scripts/run_p15.sh
cd /home/caroyf/myproject/ACMEC
a=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_main.sh)
b=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_decomp.py --part shapley)
f=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_decomp.py --part shapley --fe)
c=$(sbatch --parsable slurm/gpu.sbatch scripts/p15_mix.sh)
d=$(sbatch --parsable --dependency=afterok:$a slurm/gpu.sbatch scripts/p16_calib_full.py)
e=$(sbatch --parsable --dependency=afterok:$b:$f slurm/gpu.sbatch scripts/p17_mirror_quality.py)
echo "main $a  shapley $b  shapley_fe $f  mixture $c  calib_full $d  mirror $e"
