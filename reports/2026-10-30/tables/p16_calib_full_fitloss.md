# P16 fit loss in outcome units with the full-sample calibration

| sample             | calibration           |   beta_match |   beta_se | type             |   fit_gap_util |   fit_loss_pct_children180 |   ci95_lo |   ci95_hi |
|:-------------------|:----------------------|-------------:|----------:|:-----------------|---------------:|---------------------------:|----------:|----------:|
| all plan A         | month+modality        |     0.01386  |  0.001191 | both             |          0     |                     0      |   0       |    0      |
| all plan A         | month+modality        |     0.01386  |  0.001191 | fit-type         |          0     |                     0      |   0       |    0      |
| all plan A         | month+modality        |     0.01386  |  0.001191 | neither          |          0.726 |                     1.011  |   0.8403  |    1.183  |
| all plan A         | month+modality        |     0.01386  |  0.001191 | no demand signal |          0     |                     0      |   0       |    0      |
| all plan A         | month+modality        |     0.01386  |  0.001191 | popularity-type  |          1.122 |                     1.567  |   1.302   |    1.834  |
| all plan A         | month+modality+author |     0.001394 |  0.00109  | both             |          0     |                     0      |  -0       |    0      |
| all plan A         | month+modality+author |     0.001394 |  0.00109  | fit-type         |          0     |                     0      |  -0       |    0      |
| all plan A         | month+modality+author |     0.001394 |  0.00109  | neither          |          0.726 |                     0.1013 |  -0.05387 |    0.2566 |
| all plan A         | month+modality+author |     0.001394 |  0.00109  | no demand signal |          0     |                     0      |  -0       |    0      |
| all plan A         | month+modality+author |     0.001394 |  0.00109  | popularity-type  |          1.122 |                     0.1565 |  -0.08323 |    0.3969 |
| with demand signal | month+modality        |     0.0173   |  0.00193  | both             |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality        |     0.0173   |  0.00193  | fit-type         |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality        |     0.0173   |  0.00193  | neither          |          0.726 |                     1.264  |   0.9859  |    1.542  |
| with demand signal | month+modality        |     0.0173   |  0.00193  | no demand signal |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality        |     0.0173   |  0.00193  | popularity-type  |          1.122 |                     1.959  |   1.528   |    2.393  |
| with demand signal | month+modality+author |     0.005248 |  0.002055 | both             |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality+author |     0.005248 |  0.002055 | fit-type         |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality+author |     0.005248 |  0.002055 | neither          |          0.726 |                     0.3818 |   0.08863 |    0.6758 |
| with demand signal | month+modality+author |     0.005248 |  0.002055 | no demand signal |          0     |                     0      |   0       |    0      |
| with demand signal | month+modality+author |     0.005248 |  0.002055 | popularity-type  |          1.122 |                     0.5906 |   0.137   |    1.046  |
