#!/bin/bash
# v2.1: rebuild v2_samples tables (bootstrap intervals) for the given configs, then the final summary.
# usage: sbatch slurm/cpu.sbatch scripts/v2_collect.sh P5 P5a ...
cd /home/caroyf/myproject/ACMEC
for t in "$@"; do python scripts/v2_samples.py --tag "$t" --part collect > /dev/null && echo "collected $t"; done
python scripts/v2_sample_summary.py
