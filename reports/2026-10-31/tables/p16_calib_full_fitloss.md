# P16 fit loss in outcome units with the full-sample calibration

| sample             | calibration           |   beta_match |   beta_se | type             |   fit_gap_util |   fit_loss_pct_children180 |   ci95_lo |   ci95_hi |
|:-------------------|:----------------------|-------------:|----------:|:-----------------|---------------:|---------------------------:|----------:|----------:|
| all plan A         | month+modality        |     0.01806  |  0.001477 | both             |         0      |                     0      |  0        |    0      |
| all plan A         | month+modality        |     0.01806  |  0.001477 | fit-type         |         0      |                     0      |  0        |    0      |
| all plan A         | month+modality        |     0.01806  |  0.001477 | neither          |         0.7789 |                     1.417  |  1.188    |    1.646  |
| all plan A         | month+modality        |     0.01806  |  0.001477 | no demand signal |         0      |                     0      |  0        |    0      |
| all plan A         | month+modality        |     0.01806  |  0.001477 | popularity-type  |         0.7746 |                     1.409  |  1.182    |    1.637  |
| all plan A         | month+modality+author |     0.002559 |  0.001326 | both             |         0      |                     0      | -0        |    0      |
| all plan A         | month+modality+author |     0.002559 |  0.001326 | fit-type         |         0      |                     0      | -0        |    0      |
| all plan A         | month+modality+author |     0.002559 |  0.001326 | neither          |         0.7789 |                     0.1995 | -0.003173 |    0.4026 |
| all plan A         | month+modality+author |     0.002559 |  0.001326 | no demand signal |         0      |                     0      | -0        |    0      |
| all plan A         | month+modality+author |     0.002559 |  0.001326 | popularity-type  |         0.7746 |                     0.1984 | -0.003155 |    0.4004 |
| with demand signal | month+modality        |     0.02102  |  0.002136 | both             |         0      |                     0      |  0        |    0      |
| with demand signal | month+modality        |     0.02102  |  0.002136 | fit-type         |         0      |                     0      |  0        |    0      |
| with demand signal | month+modality        |     0.02102  |  0.002136 | neither          |         0.7789 |                     1.651  |  1.32     |    1.983  |
| with demand signal | month+modality        |     0.02102  |  0.002136 | no demand signal |         0      |                     0      |  0        |    0      |
| with demand signal | month+modality        |     0.02102  |  0.002136 | popularity-type  |         0.7746 |                     1.641  |  1.312    |    1.972  |
| with demand signal | month+modality+author |     0.003874 |  0.002144 | both             |         0      |                     0      | -0        |    0      |
| with demand signal | month+modality+author |     0.003874 |  0.002144 | fit-type         |         0      |                     0      | -0        |    0      |
| with demand signal | month+modality+author |     0.003874 |  0.002144 | neither          |         0.7789 |                     0.3022 | -0.02563  |    0.6311 |
| with demand signal | month+modality+author |     0.003874 |  0.002144 | no demand signal |         0      |                     0      | -0        |    0      |
| with demand signal | month+modality+author |     0.003874 |  0.002144 | popularity-type  |         0.7746 |                     0.3005 | -0.02549  |    0.6276 |
