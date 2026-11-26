#!/bin/bash
# P25: rebuild the card TF-IDF with the historical card versions as extra candidate documents (same rules and
# vocabulary as scripts/v2_text.py --build); outputs to interim/p25_text, production files untouched.
# usage: sbatch --mem=256G slurm/cpu.sbatch scripts/p25_text.sh
set -e
export V2_TEXT_EXTRA=$ACMEC_DATA_ROOT/interim/card_hist_clean.parquet
export V2_TEXT_OUT=$ACMEC_DATA_ROOT/interim/p25_text
mkdir -p $V2_TEXT_OUT
python /home/caroyf/myproject/ACMEC/scripts/v2_text.py --build
