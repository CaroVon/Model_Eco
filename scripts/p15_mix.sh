#!/bin/bash
# semi-synthetic check of the latent-class decomposition, then the real-data fits
set -e
python scripts/p15_decomp.py --part mixsim --avail pop
python scripts/p15_decomp.py --part mixture --avail pop
python scripts/p15_decomp.py --part mixture --avail fit
python scripts/p15_decomp.py --part mixture --avail pop --fe
