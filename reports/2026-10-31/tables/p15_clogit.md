# P15 full conditional logit by feature group (clustered SE)

| model          | group   | param                     |       coef |   se_cl |         z |
|:---------------|:--------|:--------------------------|-----------:|--------:|----------:|
| planA          | pop     | log_deg                   |  0.3474    | 0.01408 |  24.68    |
| planA          | pop     | lrec30                    |  0.6309    | 0.01914 |  32.96    |
| planA          | pop     | major_org                 |  0.2997    | 0.04558 |   6.576   |
| planA          | base    | has_deg                   |  4.687     | 0.07885 |  59.44    |
| planA          | base    | legacy                    |  1.372     | 0.05218 |  26.29    |
| planA          | base    | log_age                   | -0.5391    | 0.02056 | -26.22    |
| planA          | qual    | bench_z                   |  0.01513   | 0.08049 |   0.188   |
| planA          | qual    | bench_na                  |  0.1756    | 0.09614 |   1.826   |
| planA          | match   | size_dist                 | -0.1733    | 0.02539 |  -6.823   |
| planA          | match   | size_na                   | -0.7799    | 0.06168 | -12.64    |
| planA          | match   | task_hist                 |  1.909     | 0.07702 |  24.79    |
| planA          | match   | lang_hist                 |  0.3603    | 0.04168 |   8.644   |
| planA          | match   | ds_lang                   |  0.4591    | 0.1222  |   3.756   |
| planA          | match   | ds_task                   |  1.245     | 0.1487  |   8.373   |
| planA          | match   | txt_sim                   |  4.559     | 0.6561  |   6.948   |
| planA          | match   | txt_na                    | -0.7431    | 0.07411 | -10.03    |
| planA          | inertia | author_family             | -0.06226   | 0.08503 |  -0.7322  |
| planA          | inertia | same_author               |  6.566     | 0.4487  |  14.63    |
| planA          | local   | lco2                      |  0.676     | 0.1355  |   4.989   |
| planA          | local   | ds_couse                  |  1.15      | 0.04026 |  28.56    |
| planA          | avail   | lquant                    |  0.09867   | 0.01289 |   7.654   |
| planA          | avail   | lib_transformers          |  0.1962    | 0.08153 |   2.406   |
| planA          | avail   | lib_peft                  | -1.763     | 0.3055  |  -5.771   |
| planA          | avail   | lib_diffusers             |  0.4424    | 0.2694  |   1.642   |
| planA          | avail   | lib_sentence-transformers |  0.8735    | 0.113   |   7.73    |
| planA + hub FE | pop     | log_deg                   |  0.1544    | 0.02368 |   6.522   |
| planA + hub FE | pop     | lrec30                    |  0.4913    | 0.01928 |  25.48    |
| planA + hub FE | pop     | major_org                 |  1.433     | 0.07243 |  19.79    |
| planA + hub FE | base    | has_deg                   |  3.87      | 0.08373 |  46.22    |
| planA + hub FE | base    | legacy                    |  1.946     | 0.08547 |  22.76    |
| planA + hub FE | base    | log_age                   | -0.6081    | 0.02202 | -27.61    |
| planA + hub FE | qual    | bench_z                   |  0.1677    | 0.1751  |   0.9574  |
| planA + hub FE | qual    | bench_na                  |  0.009653  | 0.1443  |   0.06691 |
| planA + hub FE | match   | size_dist                 | -0.2487    | 0.02568 |  -9.687   |
| planA + hub FE | match   | size_na                   | -0.9526    | 0.06227 | -15.3     |
| planA + hub FE | match   | task_hist                 |  1.859     | 0.07499 |  24.79    |
| planA + hub FE | match   | lang_hist                 |  0.2733    | 0.05106 |   5.353   |
| planA + hub FE | match   | ds_lang                   |  0.288     | 0.1256  |   2.293   |
| planA + hub FE | match   | ds_task                   |  1.502     | 0.1625  |   9.242   |
| planA + hub FE | match   | txt_sim                   |  5.013     | 0.7161  |   7       |
| planA + hub FE | match   | txt_na                    | -0.5337    | 0.0795  |  -6.713   |
| planA + hub FE | inertia | author_family             |  0.03311   | 0.09246 |   0.3581  |
| planA + hub FE | inertia | same_author               |  6.313     | 0.4394  |  14.37    |
| planA + hub FE | local   | lco2                      |  0.5775    | 0.1197  |   4.826   |
| planA + hub FE | local   | ds_couse                  |  1.123     | 0.0373  |  30.11    |
| planA + hub FE | avail   | lquant                    |  0.2415    | 0.03375 |   7.156   |
| planA + hub FE | avail   | lib_transformers          |  0.1091    | 0.1006  |   1.084   |
| planA + hub FE | avail   | lib_peft                  | -1.987     | 0.3209  |  -6.192   |
| planA + hub FE | avail   | lib_diffusers             | -0.4539    | 0.3243  |  -1.4     |
| planA + hub FE | avail   | lib_sentence-transformers |  0.5986    | 0.1577  |   3.795   |
| all            | pop     | log_deg                   |  0.3418    | 0.01381 |  24.75    |
| all            | pop     | lrec30                    |  0.5716    | 0.01676 |  34.1     |
| all            | pop     | major_org                 |  0.2402    | 0.03477 |   6.908   |
| all            | base    | has_deg                   |  4.901     | 0.0661  |  74.13    |
| all            | base    | legacy                    |  1.224     | 0.04441 |  27.56    |
| all            | base    | log_age                   | -0.4977    | 0.01946 | -25.57    |
| all            | qual    | bench_z                   |  0.0007947 | 0.06987 |   0.01137 |
| all            | qual    | bench_na                  |  0.0787    | 0.07289 |   1.08    |
| all            | match   | size_dist                 | -0.1454    | 0.01717 |  -8.468   |
| all            | match   | size_na                   | -0.2955    | 0.03962 |  -7.458   |
| all            | match   | task_hist                 |  1.04      | 0.05521 |  18.84    |
| all            | match   | lang_hist                 |  0.01198   | 0.03805 |   0.3148  |
| all            | match   | ds_lang                   |  0.4395    | 0.1017  |   4.32    |
| all            | match   | ds_task                   |  1.495     | 0.247   |   6.051   |
| all            | match   | txt_sim                   |  3.173     | 1.366   |   2.323   |
| all            | match   | txt_na                    | -0.6875    | 0.07474 |  -9.199   |
| all            | inertia | author_reuse              |  1.653     | 0.08978 |  18.41    |
| all            | inertia | author_family             |  0.8028    | 0.08756 |   9.169   |
| all            | inertia | same_author               |  5.271     | 0.345   |  15.28    |
| all            | inertia | is_last                   |  2.204     | 0.1195  |  18.44    |
| all            | local   | lco2                      |  0.1078    | 0.08336 |   1.293   |
| all            | local   | ds_couse                  |  0.9932    | 0.04587 |  21.65    |
| all            | avail   | lquant                    |  0.1127    | 0.01053 |  10.7     |
| all            | avail   | lib_transformers          |  0.3353    | 0.06929 |   4.838   |
| all            | avail   | lib_peft                  | -1.746     | 0.2896  |  -6.031   |
| all            | avail   | lib_diffusers             |  0.3176    | 0.3162  |   1.004   |
| all            | avail   | lib_sentence-transformers |  0.9817    | 0.09453 |  10.38    |
| all + hub FE   | pop     | log_deg                   |  0.2       | 0.02181 |   9.168   |
| all + hub FE   | pop     | lrec30                    |  0.4569    | 0.01487 |  30.73    |
| all + hub FE   | pop     | major_org                 |  1.258     | 0.07885 |  15.95    |
| all + hub FE   | base    | has_deg                   |  4.193     | 0.0689  |  60.86    |
| all + hub FE   | base    | legacy                    |  1.72      | 0.07353 |  23.4     |
| all + hub FE   | base    | log_age                   | -0.584     | 0.02138 | -27.31    |
| all + hub FE   | qual    | bench_z                   |  0.1721    | 0.1661  |   1.036   |
| all + hub FE   | qual    | bench_na                  | -0.05347   | 0.1389  |  -0.385   |
| all + hub FE   | match   | size_dist                 | -0.2205    | 0.01834 | -12.03    |
| all + hub FE   | match   | size_na                   | -0.5051    | 0.04809 | -10.5     |
| all + hub FE   | match   | task_hist                 |  1.018     | 0.05855 |  17.38    |
| all + hub FE   | match   | lang_hist                 | -0.004326  | 0.05282 |  -0.0819  |
| all + hub FE   | match   | ds_lang                   |  0.2319    | 0.1006  |   2.305   |
| all + hub FE   | match   | ds_task                   |  1.744     | 0.2874  |   6.068   |
| all + hub FE   | match   | txt_sim                   |  3.433     | 1.615   |   2.125   |
| all + hub FE   | match   | txt_na                    | -0.4498    | 0.09825 |  -4.578   |
| all + hub FE   | inertia | author_reuse              |  1.521     | 0.1121  |  13.56    |
| all + hub FE   | inertia | author_family             |  0.8707    | 0.1012  |   8.6     |
| all + hub FE   | inertia | same_author               |  5.086     | 0.3498  |  14.54    |
| all + hub FE   | inertia | is_last                   |  2.233     | 0.1209  |  18.47    |
| all + hub FE   | local   | lco2                      |  0.144     | 0.08614 |   1.672   |
| all + hub FE   | local   | ds_couse                  |  0.9984    | 0.04249 |  23.5     |
| all + hub FE   | avail   | lquant                    |  0.2456    | 0.03057 |   8.035   |
| all + hub FE   | avail   | lib_transformers          |  0.1279    | 0.08984 |   1.424   |
| all + hub FE   | avail   | lib_peft                  | -2.206     | 0.3425  |  -6.44    |
| all + hub FE   | avail   | lib_diffusers             | -0.4091    | 0.3751  |  -1.091   |
| all + hub FE   | avail   | lib_sentence-transformers |  0.5585    | 0.1364  |   4.095   |
