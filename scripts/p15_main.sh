#!/bin/bash
# full conditional logit, MRS, per-choice classification, then P16 welfare (needs the classification)
set -e
for p in clogit mrs classify; do python scripts/p15_decomp.py --part $p; done
python scripts/p16_welfare.py
