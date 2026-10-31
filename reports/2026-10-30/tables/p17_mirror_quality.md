# P17 how much of the hub FE is not quality: identical-weight pairs

phi from the P15 plan-A conditional logit with hub FE. s = noise-corrected Var(e)/Var(phi) from demeaned pair differences; lower bound on the non-quality share (parts shared within a pair count as quality). s_sys: publisher premium of identical-weight copies (all by unsloth) times the spread of that publisher over the hubs. Adjusted bound: partition C of p15_shapley_fe, the hub-FE part of the quality share split by s_total (linear approximation); non-quality part to popularity (ratio) or to availability (ratio_if_availability).

| quantity                                                 |      value |
|:---------------------------------------------------------|-----------:|
| pairs with both repos among the hubs                     |  15        |
| hubs with FE SE < 1.0 (of 500)                           | 447        |
| var(phi) over these hubs                                 |   1.624    |
| mean FE sampling variance                                |   0.1407   |
| publisher of the copies: unsloth; hubs from it           | 102        |
| mean copy - original phi                                 |   1.906    |
| corr(phi_a, phi_b)                                       |   0.6055   |
| s raw (no noise correction)                              |   0.08148  |
| s non-quality share of var(phi)                          |   0.007102 |
| s 95% bootstrap lo                                       |  -0.05988  |
| s 95% bootstrap hi                                       |   0.06935  |
| s_sys publisher premium share                            |   0.4312   |
| s_total = s_sys + s                                      |   0.4384   |
| s_total 95% lo                                           |   0.3255   |
| s_total 95% hi                                           |   0.5432   |
| hub-FE part of quality share                             |   0.3116   |
| adjusted upper bound (s_total): fit                      |   0.2002   |
| adjusted upper bound (s_total): popularity               |   0.411    |
| adjusted upper bound (s_total): ratio                    |   0.4871   |
| adjusted upper bound (s_total): ratio_if_availability    |   0.7295   |
| adjusted upper bound (s_total lo): fit                   |   0.2354   |
| adjusted upper bound (s_total lo): popularity            |   0.3758   |
| adjusted upper bound (s_total lo): ratio                 |   0.6263   |
| adjusted upper bound (s_total lo): ratio_if_availability |   0.8577   |
| adjusted upper bound (s_total hi): fit                   |   0.1675   |
| adjusted upper bound (s_total hi): popularity            |   0.4437   |
| adjusted upper bound (s_total hi): ratio                 |   0.3776   |
| adjusted upper bound (s_total hi): ratio_if_availability |   0.6105   |

## Pairs

| a_id                               | b_id                               |   phi_a |   phi_b |   dphi |
|:-----------------------------------|:-----------------------------------|--------:|--------:|-------:|
| Qwen/Qwen2.5-1.5B-Instruct         | unsloth/Qwen2.5-1.5B-Instruct      |   1.450 |   3.463 |  2.012 |
| meta-llama/Llama-3.2-1B-Instruct   | unsloth/Llama-3.2-1B-Instruct      |   0.437 |   2.511 |  2.075 |
| Qwen/Qwen2.5-3B-Instruct           | unsloth/Qwen2.5-3B-Instruct        |   1.032 |   3.094 |  2.062 |
| meta-llama/Llama-3.2-1B            | unsloth/Llama-3.2-1B               |   0.964 |   2.656 |  1.692 |
| meta-llama/Llama-3.2-3B-Instruct   | unsloth/Llama-3.2-3B-Instruct      |   0.596 |   2.011 |  1.414 |
| Qwen/Qwen2.5-0.5B-Instruct         | unsloth/Qwen2.5-0.5B-Instruct      |   1.251 |   3.878 |  2.626 |
| Qwen/Qwen2.5-0.5B                  | unsloth/Qwen2.5-0.5B               |   1.133 |   3.089 |  1.956 |
| Qwen/Qwen2.5-1.5B                  | unsloth/Qwen2.5-1.5B               |   0.794 |   2.499 |  1.704 |
| meta-llama/Llama-3.2-3B            | unsloth/Llama-3.2-3B               |  -0.403 |   2.295 |  2.698 |
| Qwen/Qwen2.5-Coder-7B-Instruct     | unsloth/Qwen2.5-Coder-7B-Instruct  |   0.765 |   2.438 |  1.673 |
| google/gemma-3-1b-it               | unsloth/gemma-3-1b-it              |   1.716 |   2.525 |  0.808 |
| Qwen/Qwen2.5-Math-1.5B             | unsloth/Qwen2.5-Math-1.5B          |   1.432 |   2.890 |  1.458 |
| meta-llama/Llama-3.1-8B-Instruct   | unsloth/Meta-Llama-3.1-8B-Instruct |   0.539 |   2.139 |  1.600 |
| HuggingFaceTB/SmolLM-135M-Instruct | unsloth/SmolLM-135M-Instruct       |   1.701 |   3.859 |  2.158 |
| Qwen/Qwen2.5-Coder-32B-Instruct    | unsloth/Qwen2.5-Coder-32B-Instruct |   0.183 |   2.829 |  2.646 |
