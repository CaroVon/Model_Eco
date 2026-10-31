# P15 fit, holdout and drop-one-group likelihood-ratio tests

| model          | test            |         ll |   k |         ll0 |   pseudo_r2 |   hold_mean_ll |   hold_top1 |   hold_top10 |      lr_stat |           p |
|:---------------|:----------------|-----------:|----:|------------:|------------:|---------------:|------------:|-------------:|-------------:|------------:|
| planA          | full vs uniform | -4.707e+04 |  25 |  -1.252e+05 |      0.6241 |         -3.886 |      0.1337 |       0.3432 |  nan         | nan         |
| planA          | drop pop        | -6.104e+04 |  22 | nan         |    nan      |        nan     |    nan      |     nan      |    2.795e+04 |   0         |
| planA          | drop base       | -5.515e+04 |  22 | nan         |    nan      |        nan     |    nan      |     nan      |    1.617e+04 |   0         |
| planA          | drop qual       | -4.707e+04 |  23 | nan         |    nan      |        nan     |    nan      |     nan      |    3.856     |   0.1455    |
| planA          | drop match      | -4.86e+04  |  17 | nan         |    nan      |        nan     |    nan      |     nan      | 3071         |   0         |
| planA          | drop inertia    | -4.86e+04  |  23 | nan         |    nan      |        nan     |    nan      |     nan      | 3056         |   0         |
| planA          | drop local      | -4.857e+04 |  23 | nan         |    nan      |        nan     |    nan      |     nan      | 3013         |   0         |
| planA          | drop avail      | -4.723e+04 |  20 | nan         |    nan      |        nan     |    nan      |     nan      |  316.3       |   3.14e-66  |
| planA + hub FE | full vs uniform | -4.381e+04 | 525 |  -1.252e+05 |      0.6502 |         -4.039 |      0.106  |       0.3303 |  nan         | nan         |
| all            | full vs uniform | -5.571e+04 |  27 |  -1.709e+05 |      0.674  |         -2.447 |      0.4729 |       0.6662 |  nan         | nan         |
| all            | drop pop        | -6.933e+04 |  24 | nan         |    nan      |        nan     |    nan      |     nan      |    2.723e+04 |   0         |
| all            | drop base       | -6.539e+04 |  24 | nan         |    nan      |        nan     |    nan      |     nan      |    1.937e+04 |   0         |
| all            | drop qual       | -5.571e+04 |  25 | nan         |    nan      |        nan     |    nan      |     nan      |    1.038     |   0.5952    |
| all            | drop match      | -5.654e+04 |  19 | nan         |    nan      |        nan     |    nan      |     nan      | 1651         |   0         |
| all            | drop inertia    | -6.537e+04 |  23 | nan         |    nan      |        nan     |    nan      |     nan      |    1.931e+04 |   0         |
| all            | drop local      | -5.692e+04 |  25 | nan         |    nan      |        nan     |    nan      |     nan      | 2415         |   0         |
| all            | drop avail      | -5.593e+04 |  22 | nan         |    nan      |        nan     |    nan      |     nan      |  441.9       |   2.744e-93 |
| all + hub FE   | full vs uniform | -5.273e+04 | 527 |  -1.709e+05 |      0.6914 |         -2.511 |      0.4126 |       0.6286 |  nan         | nan         |
