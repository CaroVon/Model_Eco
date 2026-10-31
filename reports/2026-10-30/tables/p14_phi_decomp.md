# P14 hub fixed effects: what part is benchmark quality?

Hubs with FE: 500; with bench_z: 9. phi from the popularity mode of the 5-mode mixture (gamma = 0.5, hub FE); OLS with modality dummies.

## Share of var(phi) (Shapley over groups, modality first)

| component       |   share_var_phi |
|:----------------|----------------:|
| modality        |           0.033 |
| quality (bench) |           0.004 |
| size            |           0.001 |
| brand           |           0.003 |
| unexplained     |           0.959 |

## Coefficients (HC1 SE)

|            |   coef |    se |
|:-----------|-------:|------:|
| bench_z    |  1.140 | 1.180 |
| bench_na   | -0.051 | 0.520 |
| log_params |  0.025 | 0.081 |
| params_na  | -0.121 | 0.221 |
| major_org  |  0.200 | 0.160 |

Attribution for P15: the benchmark share counts as general quality; brand as popularity; size and the unexplained remainder are not attributable and bound the fit share (all to popularity = lower bound, all to fit = upper bound).
