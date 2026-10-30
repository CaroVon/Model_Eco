# P5 model ladder (layout of Overgoor et al. 2020, tables 3-4)

| | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Has degree | 6.873* | 4.813* | 5.255* | 4.796* | 4.739* | 4.096* | 4.056* |
| | (0.049) | (0.053) | (0.048) | (0.050) | (0.053) | (0.059) | (0.076) |
| log Degree (α) |  | 0.853* | 0.933* | 0.924* | 0.738* | 0.689* | 0.188* |
| |  | (0.005) | (0.006) | (0.008) | (0.017) | (0.017) | (0.022) |
| log(1 + age) |  |  | -0.869* | -0.885* | -1.029* | -1.058* | -0.618* |
| |  |  | (0.015) | (0.016) | (0.034) | (0.034) | (0.037) |
| Legacy (≤ T0) |  |  | 1.163* | 1.104* | 1.587* | 1.589* | 1.424* |
| |  |  | (0.049) | (0.047) | (0.096) | (0.084) | (0.082) |
| Age < 30d |  |  |  |  | -0.967* | -0.995* | -0.344* |
| |  |  |  |  | (0.111) | (0.109) | (0.113) |
| Age 30–90d |  |  |  |  | -0.710* | -0.741* | -0.554* |
| |  |  |  |  | (0.076) | (0.074) | (0.078) |
| Age 90–365d |  |  |  |  | -0.457* | -0.450* | -0.475* |
| |  |  |  |  | (0.050) | (0.048) | (0.051) |
| log(1 + children, 30d) |  |  |  |  |  |  | 0.564* |
| |  |  |  |  |  |  | (0.019) |
| Same author |  |  |  |  |  |  | 5.364* |
| |  |  |  |  |  |  | (0.271) |
| Author used this parent |  |  |  |  |  |  | 2.058* |
| |  |  |  |  |  |  | (0.099) |
| Author used this family |  |  |  |  |  |  | 1.500* |
| |  |  |  |  |  |  | (0.091) |
| log(1 + family births, 30d) |  |  |  |  |  |  | -0.090* |
| |  |  |  |  |  |  | (0.011) |
| log(1 + quantized children) |  |  |  |  |  |  | 0.251* |
| |  |  |  |  |  |  | (0.031) |
| Static features (M4, M6) |  |  |  | yes |  | yes | yes |
| Hub fixed effects (K = 500) |  |  |  |  | yes | yes | yes |
| Observations | 38,038 | 38,038 | 38,038 | 38,038 | 38,038 | 38,038 | 38,038 |
| Log-likelihood (weighted) | -112,547 | -76,846 | -72,163 | -71,118 | -67,254 | -66,459 | -55,508 |
| Holdout mean log-likelihood | -7.294 | -4.685 | -4.353 | -4.289 | -4.494 | -4.474 | -2.811 |
| Holdout top-1 accuracy | 0.009 | 0.135 | 0.131 | 0.136 | 0.141 | 0.138 | 0.326 |
| Holdout top-10 accuracy | 0.088 | 0.272 | 0.277 | 0.299 | 0.296 | 0.297 | 0.592 |

Note: conditional logit with stratified negative sampling and log q correction; choice weights n^-0.5; SE clustered by child author; * p < 0.01. Holdout = last 10% of choices in time. M0 (uniform): holdout mean log-likelihood -11.214. Coefficients are conditional associations.
