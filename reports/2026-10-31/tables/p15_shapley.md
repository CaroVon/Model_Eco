# P15 Shapley shares of the log-likelihood gain over uniform attachment

| sample   | partition                                   | group                   |   shapley_ll |     share |   total_gain |
|:---------|:--------------------------------------------|:------------------------|-------------:|----------:|-------------:|
| planA    | seven groups                                | popularity              |    3.347e+04 | 0.4282    |    7.815e+04 |
| planA    | seven groups                                | visibility / life cycle |    2.147e+04 | 0.2747    |    7.815e+04 |
| planA    | seven groups                                | general quality         |   35.15      | 0.0004497 |    7.815e+04 |
| planA    | seven groups                                | demand match            | 4223         | 0.05403   |    7.815e+04 |
| planA    | seven groups                                | inertia                 | 1954         | 0.025     |    7.815e+04 |
| planA    | seven groups                                | local search            | 4223         | 0.05404   |    7.815e+04 |
| planA    | seven groups                                | availability            |    1.278e+04 | 0.1635    |    7.815e+04 |
| planA    | A: visibility + availability as popularity  | popularity              |    6.524e+04 | 0.8347    |    7.815e+04 |
| planA    | A: visibility + availability as popularity  | fit                     | 5749         | 0.07356   |    7.815e+04 |
| planA    | A: visibility + availability as popularity  | inertia                 | 1792         | 0.02293   |    7.815e+04 |
| planA    | A: visibility + availability as popularity  | local search            | 5376         | 0.06879   |    7.815e+04 |
| planA    | B: availability as fit, visibility separate | popularity              |    3.418e+04 | 0.4373    |    7.815e+04 |
| planA    | B: availability as fit, visibility separate | visibility              |    2.191e+04 | 0.2803    |    7.815e+04 |
| planA    | B: availability as fit, visibility separate | fit                     |    1.582e+04 | 0.2025    |    7.815e+04 |
| planA    | B: availability as fit, visibility separate | inertia                 | 1958         | 0.02505   |    7.815e+04 |
| planA    | B: availability as fit, visibility separate | local search            | 4288         | 0.05486   |    7.815e+04 |
| planA    | C: visibility and availability separate     | popularity              |    3.347e+04 | 0.4282    |    7.815e+04 |
| planA    | C: visibility and availability separate     | visibility              |    2.148e+04 | 0.2749    |    7.815e+04 |
| planA    | C: visibility and availability separate     | availability            |    1.278e+04 | 0.1636    |    7.815e+04 |
| planA    | C: visibility and availability separate     | fit                     | 4246         | 0.05433   |    7.815e+04 |
| planA    | C: visibility and availability separate     | inertia                 | 1955         | 0.02501   |    7.815e+04 |
| planA    | C: visibility and availability separate     | local search            | 4223         | 0.05404   |    7.815e+04 |
| all      | seven groups                                | popularity              |    4.119e+04 | 0.3576    |    1.152e+05 |
| all      | seven groups                                | visibility / life cycle |    2.724e+04 | 0.2365    |    1.152e+05 |
| all      | seven groups                                | general quality         |   42.5       | 0.000369  |    1.152e+05 |
| all      | seven groups                                | demand match            | 3411         | 0.02962   |    1.152e+05 |
| all      | seven groups                                | inertia                 |    2.348e+04 | 0.2038    |    1.152e+05 |
| all      | seven groups                                | local search            | 4798         | 0.04167   |    1.152e+05 |
| all      | seven groups                                | availability            |    1.501e+04 | 0.1303    |    1.152e+05 |
| all      | A: visibility + availability as popularity  | popularity              |    7.608e+04 | 0.6606    |    1.152e+05 |
| all      | A: visibility + availability as popularity  | fit                     | 4615         | 0.04007   |    1.152e+05 |
| all      | A: visibility + availability as popularity  | inertia                 |    2.844e+04 | 0.2469    |    1.152e+05 |
| all      | A: visibility + availability as popularity  | local search            | 6034         | 0.0524    |    1.152e+05 |
| all      | B: availability as fit, visibility separate | popularity              |    4.175e+04 | 0.3626    |    1.152e+05 |
| all      | B: availability as fit, visibility separate | visibility              |    2.763e+04 | 0.2399    |    1.152e+05 |
| all      | B: availability as fit, visibility separate | fit                     |    1.708e+04 | 0.1483    |    1.152e+05 |
| all      | B: availability as fit, visibility separate | inertia                 |    2.382e+04 | 0.2068    |    1.152e+05 |
| all      | B: availability as fit, visibility separate | local search            | 4878         | 0.04236   |    1.152e+05 |
| all      | C: visibility and availability separate     | popularity              |    4.119e+04 | 0.3576    |    1.152e+05 |
| all      | C: visibility and availability separate     | visibility              |    2.725e+04 | 0.2366    |    1.152e+05 |
| all      | C: visibility and availability separate     | availability            |    1.501e+04 | 0.1303    |    1.152e+05 |
| all      | C: visibility and availability separate     | fit                     | 3439         | 0.02986   |    1.152e+05 |
| all      | C: visibility and availability separate     | inertia                 |    2.348e+04 | 0.2039    |    1.152e+05 |
| all      | C: visibility and availability separate     | local search            | 4799         | 0.04167   |    1.152e+05 |
