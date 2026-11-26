#!/bin/bash
# Regenerate paper tables, macros and figures for every draft (usage: sbatch slurm/cpu.sbatch scripts/paper_build.sh)
set -e
cd /home/caroyf/myproject/ACMEC
python scripts/paper_tables.py
python scripts/paper_figures.py
