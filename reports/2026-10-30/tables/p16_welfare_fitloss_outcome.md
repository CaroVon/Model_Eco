# P16 fit loss in outcome units (% of children within 180 days), with and without author FE

| calibration           |   beta_match |   beta_se | type             |   fit_gap_util |   fit_loss_pct_children180 |
|:----------------------|-------------:|----------:|:-----------------|---------------:|---------------------------:|
| month+modality        |     0.01476  |  0.003321 | both             |          0     |                     0      |
| month+modality        |     0.01476  |  0.003321 | fit-type         |          0     |                     0      |
| month+modality        |     0.01476  |  0.003321 | neither          |          0.726 |                     1.077  |
| month+modality        |     0.01476  |  0.003321 | no demand signal |          0     |                     0      |
| month+modality        |     0.01476  |  0.003321 | popularity-type  |          1.122 |                     1.67   |
| month+modality+author |    -0.005404 |  0.007315 | both             |          0     |                    -0      |
| month+modality+author |    -0.005404 |  0.007315 | fit-type         |          0     |                    -0      |
| month+modality+author |    -0.005404 |  0.007315 | neither          |          0.726 |                    -0.3916 |
| month+modality+author |    -0.005404 |  0.007315 | no demand signal |          0     |                    -0      |
| month+modality+author |    -0.005404 |  0.007315 | popularity-type  |          1.122 |                    -0.6044 |
