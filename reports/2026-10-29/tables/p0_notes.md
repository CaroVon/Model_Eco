# P0 notes (synthetic validation)

Tables: `p0_synth.csv`, `p0_fitness_diag.csv`. N = 50,000 nodes, hub stratum K = 200, s1 = s0 = 24, uniform s = 50, seed 42.

1. **PA recovery: pass.** For alpha in {0, 0.5, 1} both stratified and uniform sampling recover alpha and
   has_deg within 2 SE (|z| <= 1.7).
2. **log q correction is necessary: pass.** Dropping it gives |z| of 20-360 (e.g. alpha = 1 -> 0.24).
3. **Fitness placebo: partly fails.** On a pure-fitness network (phi ~ N(0,1), true alpha = 0):
   - M2 gives a spurious alpha = 0.61 (expected).
   - M5 with FE for the top-200 hubs still gives alpha = 0.38 (plan expected ~0).
     Split: alpha on hub candidates = -0.11 (0.017), on non-hub candidates = 0.48 (0.009)
     -> the leak comes from non-hub nodes, which share one intercept but differ in fitness.
   - FE for top-K' nodes: K'=200 -> 0.38, K'=1000 -> -0.03, K'=5000 -> -0.73.
     Too few FE leave fitness in non-hubs; too many FE on low-degree nodes overshoot negative
     (incidental-parameter / selection bias: FE are estimated from the same few choices that build degree).
   - Oracle (true phi as covariate): alpha = -0.006 (0.007).
   Implication for P7: alpha in M5 is not robust to K; the K-sensitivity (plan P7.3) must be central,
   and identifying alpha from hub candidates only (alpha_hub) looks closer to the truth than pooled alpha.
