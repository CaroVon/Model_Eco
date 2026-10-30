#!/bin/bash
# All P11 scenarios x seeds 0-2 for one fitted spec. usage: bash scripts/p11_spec.sh g0_fe
spec=$1
sc="base alpha0 norecent nofeedback pop_only m3"
[[ $spec == *_fe ]] && sc="$sc noquality"
for s in $sc; do for seed in 0 1 2; do
  python scripts/p11_generative.py --spec $spec --scenario $s --seed $seed || exit 1
done; done
