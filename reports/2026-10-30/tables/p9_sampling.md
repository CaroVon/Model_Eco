# Uniform vs stratified negative sampling: alpha and degree-bin coefficients

| data                     | sampler    | model                        | param   |   coef |    se_cl |   share_cands_deg_ge10 |   median_deg |
|:-------------------------|:-----------|:-----------------------------|:--------|-------:|---------:|-----------------------:|-------------:|
| pure PA (alpha = 1)      | stratified | log-linear                   | log_deg | 1.001  | 0.002108 |                0.6771  |          nan |
| pure PA (alpha = 1)      | uniform    | log-linear                   | log_deg | 0.9942 | 0.004453 |                0.0216  |          nan |
| pure fitness (alpha = 0) | stratified | log-linear                   | log_deg | 0.6828 | 0.006973 |                0.6563  |          nan |
| pure fitness (alpha = 0) | uniform    | log-linear                   | log_deg | 0.672  | 0.006998 |                0.01217 |          nan |
| HF                       | stratified | log-linear                   | log_deg | 0.9325 | 0.006234 |                0.6538  |          nan |
| HF                       | stratified | degree bins                  | db_2    | 1.061  | 0.07417  |              nan       |            2 |
| HF                       | stratified | degree bins                  | db_3    | 1.858  | 0.06704  |              nan       |            3 |
| HF                       | stratified | degree bins                  | db_5    | 2.449  | 0.06987  |              nan       |            7 |
| HF                       | stratified | degree bins                  | db_9    | 3.315  | 0.0842   |              nan       |           12 |
| HF                       | stratified | degree bins                  | db_17   | 4.306  | 0.09064  |              nan       |           24 |
| HF                       | stratified | degree bins                  | db_33   | 5.222  | 0.0937   |              nan       |           49 |
| HF                       | stratified | degree bins                  | db_65   | 6.034  | 0.08881  |              nan       |           94 |
| HF                       | stratified | degree bins                  | db_129  | 6.433  | 0.09074  |              nan       |          182 |
| HF                       | stratified | degree bins                  | db_257  | 6.851  | 0.0896   |              nan       |          334 |
| HF                       | stratified | degree bins                  | db_513  | 7.599  | 0.09022  |              nan       |          646 |
| HF                       | stratified | degree bins                  | db_1025 | 8.679  | 0.09271  |              nan       |         2045 |
| HF                       | stratified | log-linear, degree < 33 only | log_deg | 1.332  | 0.03621  |              nan       |          nan |
| HF                       | uniform    | log-linear                   | log_deg | 1.213  | 0.02968  |                0.01918 |          nan |
| HF                       | uniform    | degree bins                  | db_2    | 1.068  | 0.1233   |              nan       |            2 |
| HF                       | uniform    | degree bins                  | db_3    | 1.825  | 0.1148   |              nan       |            3 |
| HF                       | uniform    | degree bins                  | db_5    | 2.416  | 0.122    |              nan       |            6 |
| HF                       | uniform    | degree bins                  | db_9    | 3.213  | 0.1493   |              nan       |           12 |
| HF                       | uniform    | degree bins                  | db_17   | 4.095  | 0.1536   |              nan       |           23 |
| HF                       | uniform    | degree bins                  | db_33   | 5.364  | 0.1738   |              nan       |           47 |
| HF                       | uniform    | degree bins                  | db_65   | 6.046  | 0.1647   |              nan       |           94 |
| HF                       | uniform    | degree bins                  | db_129  | 6.656  | 0.1788   |              nan       |          182 |
| HF                       | uniform    | degree bins                  | db_257  | 6.708  | 0.2107   |              nan       |          355 |
| HF                       | uniform    | degree bins                  | db_513  | 7.497  | 0.2464   |              nan       |          711 |
| HF                       | uniform    | degree bins                  | db_1025 | 8.217  | 0.2208   |              nan       |         7896 |
| HF                       | uniform    | log-linear, degree < 33 only | log_deg | 1.291  | 0.0494   |              nan       |          nan |
