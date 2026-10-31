# P15 full conditional logit by feature group (clustered SE)

| model          | group   | param                     |       coef |   se_cl |         z |
|:---------------|:--------|:--------------------------|-----------:|--------:|----------:|
| planA          | pop     | log_deg                   |  0.3413    | 0.01397 |  24.43    |
| planA          | pop     | lrec30                    |  0.6314    | 0.01914 |  33       |
| planA          | pop     | major_org                 |  0.2921    | 0.04594 |   6.358   |
| planA          | base    | has_deg                   |  4.719     | 0.072   |  65.54    |
| planA          | base    | legacy                    |  1.442     | 0.05208 |  27.7     |
| planA          | base    | log_age                   | -0.5487    | 0.02034 | -26.98    |
| planA          | qual    | bench_z                   |  0.008207  | 0.07997 |   0.1026  |
| planA          | qual    | bench_na                  |  0.1404    | 0.0961  |   1.461   |
| planA          | match   | size_dist                 | -0.1777    | 0.02569 |  -6.916   |
| planA          | match   | size_na                   | -0.8058    | 0.06266 | -12.86    |
| planA          | match   | task_hist                 |  1.888     | 0.07435 |  25.39    |
| planA          | match   | lang_hist                 |  0.357     | 0.04107 |   8.692   |
| planA          | match   | ds_lang                   |  0.4813    | 0.1239  |   3.885   |
| planA          | match   | ds_task                   |  1.204     | 0.1498  |   8.038   |
| planA          | inertia | author_family             | -0.06417   | 0.08541 |  -0.7513  |
| planA          | inertia | same_author               |  6.89      | 0.4138  |  16.65    |
| planA          | local   | lco2                      |  0.6804    | 0.1336  |   5.092   |
| planA          | local   | ds_couse                  |  1.144     | 0.04004 |  28.58    |
| planA          | avail   | lquant                    |  0.1113    | 0.01289 |   8.637   |
| planA          | avail   | lib_transformers          |  0.2091    | 0.08068 |   2.591   |
| planA          | avail   | lib_peft                  | -1.885     | 0.3018  |  -6.246   |
| planA          | avail   | lib_diffusers             |  0.4303    | 0.2663  |   1.616   |
| planA          | avail   | lib_sentence-transformers |  0.8719    | 0.1123  |   7.76    |
| planA + hub FE | pop     | log_deg                   |  0.1525    | 0.02373 |   6.426   |
| planA + hub FE | pop     | lrec30                    |  0.4913    | 0.01931 |  25.44    |
| planA + hub FE | pop     | major_org                 |  1.466     | 0.07252 |  20.22    |
| planA + hub FE | base    | has_deg                   |  3.883     | 0.07989 |  48.6     |
| planA + hub FE | base    | legacy                    |  1.966     | 0.0857  |  22.94    |
| planA + hub FE | base    | log_age                   | -0.6132    | 0.02165 | -28.32    |
| planA + hub FE | qual    | bench_z                   |  0.1635    | 0.1734  |   0.9428  |
| planA + hub FE | qual    | bench_na                  | -0.01618   | 0.1445  |  -0.112   |
| planA + hub FE | match   | size_dist                 | -0.2553    | 0.02597 |  -9.83    |
| planA + hub FE | match   | size_na                   | -0.9744    | 0.06312 | -15.44    |
| planA + hub FE | match   | task_hist                 |  1.841     | 0.07194 |  25.6     |
| planA + hub FE | match   | lang_hist                 |  0.2786    | 0.04961 |   5.616   |
| planA + hub FE | match   | ds_lang                   |  0.3483    | 0.1253  |   2.779   |
| planA + hub FE | match   | ds_task                   |  1.48      | 0.1639  |   9.032   |
| planA + hub FE | inertia | author_family             |  0.03144   | 0.09311 |   0.3376  |
| planA + hub FE | inertia | same_author               |  6.659     | 0.3931  |  16.94    |
| planA + hub FE | local   | lco2                      |  0.5819    | 0.1199  |   4.855   |
| planA + hub FE | local   | ds_couse                  |  1.118     | 0.03753 |  29.78    |
| planA + hub FE | avail   | lquant                    |  0.248     | 0.03408 |   7.278   |
| planA + hub FE | avail   | lib_transformers          |  0.1094    | 0.0991  |   1.104   |
| planA + hub FE | avail   | lib_peft                  | -2.113     | 0.3163  |  -6.68    |
| planA + hub FE | avail   | lib_diffusers             | -0.3482    | 0.319   |  -1.091   |
| planA + hub FE | avail   | lib_sentence-transformers |  0.6035    | 0.1565  |   3.856   |
| all            | pop     | log_deg                   |  0.3372    | 0.01373 |  24.55    |
| all            | pop     | lrec30                    |  0.574     | 0.01671 |  34.35    |
| all            | pop     | major_org                 |  0.2345    | 0.03519 |   6.664   |
| all            | base    | has_deg                   |  4.936     | 0.06271 |  78.71    |
| all            | base    | legacy                    |  1.272     | 0.04423 |  28.77    |
| all            | base    | log_age                   | -0.5021    | 0.01934 | -25.97    |
| all            | qual    | bench_z                   | -0.004105  | 0.06836 |  -0.06005 |
| all            | qual    | bench_na                  |  0.05326   | 0.07155 |   0.7444  |
| all            | match   | size_dist                 | -0.1484    | 0.01723 |  -8.61    |
| all            | match   | size_na                   | -0.3105    | 0.03901 |  -7.96    |
| all            | match   | task_hist                 |  1.022     | 0.05114 |  19.99    |
| all            | match   | lang_hist                 |  0.01034   | 0.03574 |   0.2894  |
| all            | match   | ds_lang                   |  0.4738    | 0.1025  |   4.621   |
| all            | match   | ds_task                   |  1.47      | 0.2509  |   5.86    |
| all            | inertia | author_reuse              |  1.653     | 0.08889 |  18.59    |
| all            | inertia | author_family             |  0.8027    | 0.08934 |   8.984   |
| all            | inertia | same_author               |  5.693     | 0.2876  |  19.8     |
| all            | inertia | is_last                   |  2.202     | 0.119   |  18.51    |
| all            | local   | lco2                      |  0.1031    | 0.08413 |   1.226   |
| all            | local   | ds_couse                  |  0.9935    | 0.04602 |  21.59    |
| all            | avail   | lquant                    |  0.121     | 0.01047 |  11.56    |
| all            | avail   | lib_transformers          |  0.3405    | 0.0691  |   4.929   |
| all            | avail   | lib_peft                  | -1.889     | 0.2896  |  -6.521   |
| all            | avail   | lib_diffusers             |  0.28      | 0.3157  |   0.8871  |
| all            | avail   | lib_sentence-transformers |  0.973     | 0.09492 |  10.25    |
| all + hub FE   | pop     | log_deg                   |  0.1982    | 0.02174 |   9.115   |
| all + hub FE   | pop     | lrec30                    |  0.4577    | 0.01493 |  30.65    |
| all + hub FE   | pop     | major_org                 |  1.285     | 0.07651 |  16.8     |
| all + hub FE   | base    | has_deg                   |  4.206     | 0.0697  |  60.35    |
| all + hub FE   | base    | legacy                    |  1.731     | 0.0748  |  23.15    |
| all + hub FE   | base    | log_age                   | -0.5824    | 0.02164 | -26.92    |
| all + hub FE   | qual    | bench_z                   |  0.1614    | 0.1581  |   1.021   |
| all + hub FE   | qual    | bench_na                  | -0.07278   | 0.1312  |  -0.5546  |
| all + hub FE   | match   | size_dist                 | -0.224     | 0.01832 | -12.22    |
| all + hub FE   | match   | size_na                   | -0.5243    | 0.04648 | -11.28    |
| all + hub FE   | match   | task_hist                 |  1.002     | 0.05406 |  18.53    |
| all + hub FE   | match   | lang_hist                 | -0.0005485 | 0.04887 |  -0.01122 |
| all + hub FE   | match   | ds_lang                   |  0.2789    | 0.1009  |   2.764   |
| all + hub FE   | match   | ds_task                   |  1.743     | 0.2954  |   5.9     |
| all + hub FE   | inertia | author_reuse              |  1.529     | 0.1106  |  13.83    |
| all + hub FE   | inertia | author_family             |  0.8642    | 0.1033  |   8.367   |
| all + hub FE   | inertia | same_author               |  5.527     | 0.2904  |  19.03    |
| all + hub FE   | inertia | is_last                   |  2.234     | 0.1214  |  18.4     |
| all + hub FE   | local   | lco2                      |  0.1408    | 0.08864 |   1.589   |
| all + hub FE   | local   | ds_couse                  |  1.001     | 0.04281 |  23.38    |
| all + hub FE   | avail   | lquant                    |  0.2486    | 0.03064 |   8.112   |
| all + hub FE   | avail   | lib_transformers          |  0.1204    | 0.08817 |   1.366   |
| all + hub FE   | avail   | lib_peft                  | -2.379     | 0.3413  |  -6.971   |
| all + hub FE   | avail   | lib_diffusers             | -0.3986    | 0.3715  |  -1.073   |
| all + hub FE   | avail   | lib_sentence-transformers |  0.5506    | 0.1355  |   4.064   |
