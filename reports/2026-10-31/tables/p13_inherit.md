# P13 inheritance check: child metadata vs chosen parent

identical / overlap: among choices where both child and parent have the field. chance: identical rate against a random parent chosen in the same modality and month. excess = identical - chance. Fields with a high excess are largely copied from the parent and cannot serve as demand signals.

| sample                      | field        |   child_has |   both_has |   identical |   overlap |   chance |   excess |   n_both |
|:----------------------------|:-------------|------------:|-----------:|------------:|----------:|---------:|---------:|---------:|
| all choices                 | pipeline_tag |       0.431 |      0.430 |       0.710 |     0.710 |    0.580 |    0.129 |   181731 |
| all choices                 | library_name |       0.948 |      0.939 |       0.495 |     0.495 |    0.480 |    0.015 |   396993 |
| all choices                 | languages    |       0.188 |      0.162 |       0.800 |     0.967 |    0.675 |    0.125 |    68483 |
| all choices                 | datasets     |       0.116 |      0.048 |       0.093 |     0.105 |    0.010 |    0.083 |    20324 |
| all choices                 | licenses     |       0.561 |      0.547 |       0.848 |     0.848 |    0.422 |    0.426 |   231202 |
| all choices                 | params_b     |       0.430 |      0.421 |       0.930 |     0.930 |    0.204 |    0.726 |   177985 |
| first author-parent choices | pipeline_tag |       0.575 |      0.573 |       0.686 |     0.686 |    0.531 |    0.155 |    76481 |
| first author-parent choices | library_name |       0.915 |      0.899 |       0.679 |     0.679 |    0.654 |    0.024 |   119969 |
| first author-parent choices | languages    |       0.264 |      0.227 |       0.751 |     0.955 |    0.609 |    0.142 |    30307 |
| first author-parent choices | datasets     |       0.188 |      0.079 |       0.130 |     0.146 |    0.007 |    0.123 |    10512 |
| first author-parent choices | licenses     |       0.745 |      0.726 |       0.832 |     0.833 |    0.406 |    0.426 |    96875 |
| first author-parent choices | params_b     |       0.257 |      0.246 |       0.858 |     0.858 |    0.132 |    0.726 |    32826 |
