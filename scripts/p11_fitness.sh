#!/bin/bash
# P11 fitness spec: alpha counterfactuals x seeds 0-2, then collect. usage: bash scripts/p11_fitness.sh
for s in fit_a0 fit_a0.25 fit_a0.5 fit_a1 pop_only; do for seed in 0 1 2; do
  python scripts/p11_generative.py --spec fitness --scenario $s --seed $seed || exit 1
done; done
python scripts/p11_generative.py --collect
