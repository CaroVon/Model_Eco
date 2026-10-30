# P0b notes (identification study, 2026-10-28)

Tables: `p0b_alpha.md` (alpha by estimator x scenario, mean of seeds 42-44), `p0b_all.csv` (all params).
Synthetic networks: N = 50,000, hub K = 200, split-sample FE K' = 1000; truth alpha = 0.5 (pa, mix*) or 0 (fit*).

1. No estimator passes. Every estimator either absorbs latent quality into alpha (+0.3 to +0.6 when
   quality varies) or is biased when there is no quality variation:
   - M2: correct only without quality (pa 0.507).
   - M5 (hubs by final degree): biased even under pure PA (0.426): hub selection looks ahead.
   - M5_EXANTE / SPLIT_COV: fix the look-ahead (pa 0.509 / 0.511) but do not control quality
     (fit 0.82 / 0.80). The split-sample FE carry almost no information (coef on phi_hat ~ 0).
   - SPLIT_OFF: near 0 on pure fitness (0.043) but also near 0 under true PA (pa -0.041): it erases PA.
   - HUB (within-hub): about -0.2 to -0.75 off; invalid.
   - RECENT: recent-activity proxies get ~0 coefficients; identical to M5.
2. Quality decay (fit_decay, mix_decay) makes every estimator worse.
3. Bulk uploaders add a further +0.1 to +0.4 to alpha. Down-weighting by n^-gamma removes most of that:
   M5 fit 0.548 -> fit_bulk 0.924 (g=0), 0.568 (g=0.5), 0.417 (g=1); mix 0.790 -> mix_bulk 0.853, 0.710, 0.682.
   gamma = 0.5 roughly restores the no-bulk value -> supports gamma = 0.5 as the main weighting.
   author_reuse as a control does not fix it.
Implication: with one cumulative-degree regressor, quality and PA are not separated by FE on a subset of
nodes. Next candidates (P0c): node-level FE for all nodes with >= m children + split-panel jackknife
bias correction (incidental-parameter bias is O(1/T)); growth-curvature moments (indirect inference);
exogenous shocks to visibility in the real data.

# P0c / P0d notes (2026-10-29)
Tables: p0c_summary.md, p0d_summary.md, p0d_all.csv. Count panel with node FE for every node.
- P0c: Poisson FE MLE has large negative bias (pa -1.27); split-panel jackknife does not fix it (pa 0.04).
  Just-identified quasi-differenced GMM is roughly unbiased without bulk uploaders but noisy.
- P0d (150k nodes, 5 seeds), over-identified two-step GMM (QD_GMM2), period 200 days, all choices:
  pa 0.500 (sd 0.10), fit -0.069 (0.09), mix 0.582 (0.16), fit_decay -0.023 (0.04), mix_decay 0.406 (0.02).
  First estimator that passes all no-bulk scenarios within ~1 SE. Clustered SEs agree with the seed sd.
  Numerical failure: 5 of 90 QD_GMM2 fits diverged (mix seeds 43/46, fit_bulk seed 42 with all choices;
  fit_bulk seeds 42/46 with n^-0.5 weights); excluded from the numbers above -> needs safeguards.
- Bulk uploaders remain unsolved: all choices -> 2.0 (fit_bulk), -0.43 (mix_bulk); n^-0.5 weights invalid
  under GMM; first-choice-only outcome closest (fit_bulk 0.05, mix_bulk 0.75 at 200 days) but SE 0.4-0.6.

# P0e / P0f / P5 rounds 1-3 (2026-10-29)
- P0e (launch hype: +2 for age < 30d, +1 for 30-90d): without age terms alpha is strongly negative
  (pa_hype -1.20, fit_hype -1.50, mix_hype -0.68). Bins matching the truth (<30, 30-90, 90-365) recover it
  (0.67 / -0.12 / 0.54, SE 0.15-0.40, J passes); log(age) is wrong and J flags it (J in the hundreds).
- P0f: finer bins (0-7, 7-30, 30-90, 90-180 vs >= 180, 20 overidentifying df) are biased again
  (pa_hype ~0.15, fit_hype -0.51, mix_hype 0.21): likely many/weak-instrument bias toward the inconsistent
  Poisson FE. Family x period test is void in the simulation (synthetic network is one tree).
- Real data: round 1 (no age) alpha -0.10, J 96/8; has_deg -1.1 is driven by self-choices (3.3% of choices).
  Round 2 (age 3 bins, no self) alpha -0.10 (0.03), J 148/17. Round 3 (age 4 bins): modality groups 0.008
  (0.034) J 123/20; top-100 family x period -0.078 (0.054) J 85/20; top-300 -0.26; top-1000 -0.96 (J 857).
  No specification passes Hansen J; alpha ranges about -0.26..+0.05 across defensible specs.
  J by family: largest contributions are big hubs (Qwen2.5-0.5B/1.5B, Llama-3.1-8B, Mistral-7B, Llama-3-8B,
  SDXL) but each <= 1.2 of 85 -> violation is diffuse, not a few families.
