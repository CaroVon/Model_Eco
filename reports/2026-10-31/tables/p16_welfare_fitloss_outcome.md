# P16 fit loss in outcome units (% of children within 180 days), with and without author FE

| calibration           |   beta_match |   beta_se | type             |   fit_gap_util |   fit_loss_pct_children180 |
|:----------------------|-------------:|----------:|:-----------------|---------------:|---------------------------:|
| month+modality        |      0.02128 |  0.004527 | both             |         0      |                      0     |
| month+modality        |      0.02128 |  0.004527 | fit-type         |         0      |                      0     |
| month+modality        |      0.02128 |  0.004527 | neither          |         0.7789 |                      1.671 |
| month+modality        |      0.02128 |  0.004527 | no demand signal |         0      |                      0     |
| month+modality        |      0.02128 |  0.004527 | popularity-type  |         0.7746 |                      1.662 |
| month+modality+author |     -0.02101 |  0.009828 | both             |         0      |                     -0     |
| month+modality+author |     -0.02101 |  0.009828 | fit-type         |         0      |                     -0     |
| month+modality+author |     -0.02101 |  0.009828 | neither          |         0.7789 |                     -1.623 |
| month+modality+author |     -0.02101 |  0.009828 | no demand signal |         0      |                     -0     |
| month+modality+author |     -0.02101 |  0.009828 | popularity-type  |         0.7746 |                     -1.614 |
