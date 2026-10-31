# P15 Shapley shares of the log-likelihood gain over uniform attachment

| sample   | partition                                   | group                   |   shapley_ll |     share |   total_gain |
|:---------|:--------------------------------------------|:------------------------|-------------:|----------:|-------------:|
| planA    | seven groups                                | popularity              |    3.346e+04 | 0.4296    |    7.788e+04 |
| planA    | seven groups                                | visibility / life cycle |    2.172e+04 | 0.2789    |    7.788e+04 |
| planA    | seven groups                                | general quality         |   35.29      | 0.0004531 |    7.788e+04 |
| planA    | seven groups                                | demand match            | 3488         | 0.04478   |    7.788e+04 |
| planA    | seven groups                                | inertia                 | 2088         | 0.02681   |    7.788e+04 |
| planA    | seven groups                                | local search            | 4222         | 0.0542    |    7.788e+04 |
| planA    | seven groups                                | availability            |    1.287e+04 | 0.1652    |    7.788e+04 |
| planA    | A: visibility + availability as popularity  | popularity              |    6.572e+04 | 0.8438    |    7.788e+04 |
| planA    | A: visibility + availability as popularity  | fit                     | 4872         | 0.06256   |    7.788e+04 |
| planA    | A: visibility + availability as popularity  | inertia                 | 1908         | 0.0245    |    7.788e+04 |
| planA    | A: visibility + availability as popularity  | local search            | 5381         | 0.06909   |    7.788e+04 |
| planA    | B: availability as fit, visibility separate | popularity              |    3.411e+04 | 0.438     |    7.788e+04 |
| planA    | B: availability as fit, visibility separate | visibility              |    2.21e+04  | 0.2838    |    7.788e+04 |
| planA    | B: availability as fit, visibility separate | fit                     |    1.529e+04 | 0.1963    |    7.788e+04 |
| planA    | B: availability as fit, visibility separate | inertia                 | 2094         | 0.02689   |    7.788e+04 |
| planA    | B: availability as fit, visibility separate | local search            | 4283         | 0.05499   |    7.788e+04 |
| planA    | C: visibility and availability separate     | popularity              |    3.346e+04 | 0.4296    |    7.788e+04 |
| planA    | C: visibility and availability separate     | visibility              |    2.173e+04 | 0.279     |    7.788e+04 |
| planA    | C: visibility and availability separate     | availability            |    1.287e+04 | 0.1653    |    7.788e+04 |
| planA    | C: visibility and availability separate     | fit                     | 3512         | 0.04509   |    7.788e+04 |
| planA    | C: visibility and availability separate     | inertia                 | 2089         | 0.02682   |    7.788e+04 |
| planA    | C: visibility and availability separate     | local search            | 4222         | 0.05421   |    7.788e+04 |
| all      | seven groups                                | popularity              |    4.121e+04 | 0.3585    |    1.15e+05  |
| all      | seven groups                                | visibility / life cycle |    2.752e+04 | 0.2393    |    1.15e+05  |
| all      | seven groups                                | general quality         |   43         | 0.000374  |    1.15e+05  |
| all      | seven groups                                | demand match            | 2636         | 0.02293   |    1.15e+05  |
| all      | seven groups                                | inertia                 |    2.366e+04 | 0.2058    |    1.15e+05  |
| all      | seven groups                                | local search            | 4794         | 0.0417    |    1.15e+05  |
| all      | seven groups                                | availability            |    1.51e+04  | 0.1313    |    1.15e+05  |
| all      | A: visibility + availability as popularity  | popularity              |    7.661e+04 | 0.6664    |    1.15e+05  |
| all      | A: visibility + availability as popularity  | fit                     | 3685         | 0.03206   |    1.15e+05  |
| all      | A: visibility + availability as popularity  | inertia                 |    2.864e+04 | 0.2491    |    1.15e+05  |
| all      | A: visibility + availability as popularity  | local search            | 6034         | 0.05248   |    1.15e+05  |
| all      | B: availability as fit, visibility separate | popularity              |    4.172e+04 | 0.3629    |    1.15e+05  |
| all      | B: availability as fit, visibility separate | visibility              |    2.784e+04 | 0.2422    |    1.15e+05  |
| all      | B: availability as fit, visibility separate | fit                     |    1.654e+04 | 0.1438    |    1.15e+05  |
| all      | B: availability as fit, visibility separate | inertia                 |    2.399e+04 | 0.2087    |    1.15e+05  |
| all      | B: availability as fit, visibility separate | local search            | 4871         | 0.04237   |    1.15e+05  |
| all      | C: visibility and availability separate     | popularity              |    4.121e+04 | 0.3585    |    1.15e+05  |
| all      | C: visibility and availability separate     | visibility              |    2.753e+04 | 0.2394    |    1.15e+05  |
| all      | C: visibility and availability separate     | availability            |    1.51e+04  | 0.1314    |    1.15e+05  |
| all      | C: visibility and availability separate     | fit                     | 2665         | 0.02318   |    1.15e+05  |
| all      | C: visibility and availability separate     | inertia                 |    2.366e+04 | 0.2058    |    1.15e+05  |
| all      | C: visibility and availability separate     | local search            | 4795         | 0.04171   |    1.15e+05  |
