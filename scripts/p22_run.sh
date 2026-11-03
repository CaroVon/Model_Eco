#!/bin/bash
# P22 chain. usage: bash scripts/p22_run.sh   (stages and environments: see scripts/p22_embed.py)
cd /home/caroyf/myproject/ACMEC
a=$(sbatch --parsable slurm/cpu.sbatch scripts/p22_embed.py --prep)
b=$(sbatch --parsable --dependency=afterok:$a slurm/gpu_vllm.sbatch scripts/p22_embed.py --embed)
c=$(sbatch --parsable --dependency=afterok:$b slurm/gpu.sbatch scripts/p22_embed.py --build)
d=$(sbatch --parsable --dependency=afterok:$a slurm/cpu.sbatch scripts/p22_embed.py --label-prep)
e=$(sbatch --parsable --dependency=afterok:$d slurm/gpu_vllm.sbatch scripts/p22_embed.py --label)
f=$(sbatch --parsable --dependency=afterok:$e slurm/cpu.sbatch scripts/p22_embed.py --label-eval)
echo "prep $a embed $b build $c label-prep $d label $e eval $f"
