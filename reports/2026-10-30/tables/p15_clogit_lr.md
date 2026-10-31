# P15 fit, holdout and drop-one-group likelihood-ratio tests

| model          | test            |         ll |   k |         ll0 |   pseudo_r2 |   hold_mean_ll |   hold_top1 |   hold_top10 |      lr_stat |            p |
|:---------------|:----------------|-----------:|----:|------------:|------------:|---------------:|------------:|-------------:|-------------:|-------------:|
| planA          | full vs uniform | -4.734e+04 |  23 |  -1.252e+05 |      0.622  |         -3.931 |      0.1279 |       0.3246 |  nan         | nan          |
| planA          | drop pop        | -6.12e+04  |  20 | nan         |    nan      |        nan     |    nan      |     nan      |    2.772e+04 |   0          |
| planA          | drop base       | -5.58e+04  |  20 | nan         |    nan      |        nan     |    nan      |     nan      |    1.692e+04 |   0          |
| planA          | drop qual       | -4.734e+04 |  21 | nan         |    nan      |        nan     |    nan      |     nan      |    2.437     |   0.2956     |
| planA          | drop match      | -4.86e+04  |  17 | nan         |    nan      |        nan     |    nan      |     nan      | 2528         |   0          |
| planA          | drop inertia    | -4.912e+04 |  21 | nan         |    nan      |        nan     |    nan      |     nan      | 3566         |   0          |
| planA          | drop local      | -4.885e+04 |  21 | nan         |    nan      |        nan     |    nan      |     nan      | 3012         |   0          |
| planA          | drop avail      | -4.753e+04 |  18 | nan         |    nan      |        nan     |    nan      |     nan      |  371.1       |   4.9e-78    |
| planA + hub FE | full vs uniform | -4.403e+04 | 523 |  -1.252e+05 |      0.6484 |         -4.043 |      0.1022 |       0.322  |  nan         | nan          |
| all            | full vs uniform | -5.591e+04 |  25 |  -1.709e+05 |      0.6728 |         -2.468 |      0.4705 |       0.6584 |  nan         | nan          |
| all            | drop pop        | -6.948e+04 |  22 | nan         |    nan      |        nan     |    nan      |     nan      |    2.713e+04 |   0          |
| all            | drop base       | -6.599e+04 |  22 | nan         |    nan      |        nan     |    nan      |     nan      |    2.015e+04 |   0          |
| all            | drop qual       | -5.591e+04 |  23 | nan         |    nan      |        nan     |    nan      |     nan      |    0.4678    |   0.7915     |
| all            | drop match      | -5.654e+04 |  19 | nan         |    nan      |        nan     |    nan      |     nan      | 1245         |   8.157e-266 |
| all            | drop inertia    | -6.593e+04 |  21 | nan         |    nan      |        nan     |    nan      |     nan      |    2.003e+04 |   0          |
| all            | drop local      | -5.712e+04 |  23 | nan         |    nan      |        nan     |    nan      |     nan      | 2411         |   0          |
| all            | drop avail      | -5.616e+04 |  20 | nan         |    nan      |        nan     |    nan      |     nan      |  498.7       |   1.491e-105 |
| all + hub FE   | full vs uniform | -5.287e+04 | 525 |  -1.709e+05 |      0.6906 |         -2.526 |      0.4131 |       0.6317 |  nan         | nan          |
