# P13 demand-match features (10% sample)

Coverage = share of choices with author history / usable dataset profile. Means over candidate rows (chosen vs not chosen; negative-sampled rows, not weighted by log q).

| sample                      | feature     |   choices_with_signal_hist |   choices_with_signal_ds |   mean_chosen |   mean_other |
|:----------------------------|:------------|---------------------------:|-------------------------:|--------------:|-------------:|
| all choices                 | hist_has    |                      0.836 |                    0.096 |         0.678 |        0.864 |
| all choices                 | size_dist   |                      0.836 |                    0.096 |         0.427 |        0.554 |
| all choices                 | size_na     |                      0.836 |                    0.096 |         0.530 |        0.554 |
| all choices                 | task_hist   |                      0.836 |                    0.096 |         0.622 |        0.602 |
| all choices                 | lang_hist   |                      0.836 |                    0.096 |         0.430 |        0.504 |
| all choices                 | ds_has      |                      0.836 |                    0.096 |         0.096 |        0.082 |
| all choices                 | ds_lang     |                      0.836 |                    0.096 |         0.027 |        0.021 |
| all choices                 | ds_task     |                      0.836 |                    0.096 |         0.055 |        0.036 |
| all choices                 | ds_couse    |                      0.836 |                    0.096 |         0.142 |        0.005 |
| all choices                 | org_noncomm |                      0.836 |                    0.096 |         0.000 |        0.000 |
| first author-parent choices | hist_has    |                      0.480 |                    0.153 |         0.480 |        0.547 |
| first author-parent choices | size_dist   |                      0.480 |                    0.153 |         0.238 |        0.321 |
| first author-parent choices | size_na     |                      0.480 |                    0.153 |         0.739 |        0.726 |
| first author-parent choices | task_hist   |                      0.480 |                    0.153 |         0.392 |        0.365 |
| first author-parent choices | lang_hist   |                      0.480 |                    0.153 |         0.293 |        0.317 |
| first author-parent choices | ds_has      |                      0.480 |                    0.153 |         0.153 |        0.133 |
| first author-parent choices | ds_lang     |                      0.480 |                    0.153 |         0.042 |        0.039 |
| first author-parent choices | ds_task     |                      0.480 |                    0.153 |         0.084 |        0.057 |
| first author-parent choices | ds_couse    |                      0.480 |                    0.153 |         0.279 |        0.008 |
| first author-parent choices | org_noncomm |                      0.480 |                    0.153 |         0.000 |        0.000 |
