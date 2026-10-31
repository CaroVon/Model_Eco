# P17 how much of the hub FE is not quality: identical-weight pairs

phi from the P15 plan-A conditional logit with hub FE. s = noise-corrected Var(e)/Var(phi) from demeaned pair differences; lower bound on the non-quality share (parts shared within a pair count as quality). s_sys: publisher premium of identical-weight copies (all by unsloth) times the spread of that publisher over the hubs. Adjusted bound: partition C of p15_shapley_fe, the hub-FE part of the quality share split by s_total (linear approximation); non-quality part to popularity (ratio) or to availability (ratio_if_availability).

| quantity                                                 |     value |
|:---------------------------------------------------------|----------:|
| pairs with both repos among the hubs                     |  15       |
| hubs with FE SE < 1.0 (of 500)                           | 452       |
| var(phi) over these hubs                                 |   1.614   |
| mean FE sampling variance                                |   0.1507  |
| publisher of the copies: unsloth; hubs from it           | 102       |
| mean copy - original phi                                 |   1.84    |
| corr(phi_a, phi_b)                                       |   0.5714  |
| s raw (no noise correction)                              |   0.09127 |
| s non-quality share of var(phi)                          |   0.01949 |
| s 95% bootstrap lo                                       |  -0.03399 |
| s 95% bootstrap hi                                       |   0.07135 |
| s_sys publisher premium share                            |   0.4043  |
| s_total = s_sys + s                                      |   0.4238  |
| s_total 95% lo                                           |   0.3033  |
| s_total 95% hi                                           |   0.5307  |
| hub-FE part of quality share                             |   0.3097  |
| adjusted upper bound (s_total): fit                      |   0.2101  |
| adjusted upper bound (s_total): popularity               |   0.4047  |
| adjusted upper bound (s_total): ratio                    |   0.519   |
| adjusted upper bound (s_total): ratio_if_availability    |   0.768   |
| adjusted upper bound (s_total lo): fit                   |   0.2474  |
| adjusted upper bound (s_total lo): popularity            |   0.3674  |
| adjusted upper bound (s_total lo): ratio                 |   0.6734  |
| adjusted upper bound (s_total lo): ratio_if_availability |   0.9045  |
| adjusted upper bound (s_total hi): fit                   |   0.177   |
| adjusted upper bound (s_total hi): popularity            |   0.4378  |
| adjusted upper bound (s_total hi): ratio                 |   0.4042  |
| adjusted upper bound (s_total hi): ratio_if_availability |   0.647   |

## Pairs

| a_id                               | b_id                               |   phi_a |   phi_b |   dphi |
|:-----------------------------------|:-----------------------------------|--------:|--------:|-------:|
| Qwen/Qwen2.5-1.5B-Instruct         | unsloth/Qwen2.5-1.5B-Instruct      |   1.615 |   3.431 |  1.816 |
| meta-llama/Llama-3.2-1B-Instruct   | unsloth/Llama-3.2-1B-Instruct      |   0.461 |   2.492 |  2.031 |
| Qwen/Qwen2.5-3B-Instruct           | unsloth/Qwen2.5-3B-Instruct        |   1.186 |   3.072 |  1.886 |
| meta-llama/Llama-3.2-1B            | unsloth/Llama-3.2-1B               |   0.985 |   2.632 |  1.647 |
| meta-llama/Llama-3.2-3B-Instruct   | unsloth/Llama-3.2-3B-Instruct      |   0.625 |   1.988 |  1.363 |
| Qwen/Qwen2.5-0.5B-Instruct         | unsloth/Qwen2.5-0.5B-Instruct      |   1.400 |   3.844 |  2.444 |
| Qwen/Qwen2.5-0.5B                  | unsloth/Qwen2.5-0.5B               |   1.273 |   3.172 |  1.899 |
| Qwen/Qwen2.5-1.5B                  | unsloth/Qwen2.5-1.5B               |   0.935 |   2.561 |  1.626 |
| meta-llama/Llama-3.2-3B            | unsloth/Llama-3.2-3B               |  -0.376 |   2.276 |  2.652 |
| Qwen/Qwen2.5-Coder-7B-Instruct     | unsloth/Qwen2.5-Coder-7B-Instruct  |   0.796 |   2.389 |  1.593 |
| google/gemma-3-1b-it               | unsloth/gemma-3-1b-it              |   1.699 |   2.493 |  0.795 |
| Qwen/Qwen2.5-Math-1.5B             | unsloth/Qwen2.5-Math-1.5B          |   1.577 |   2.736 |  1.159 |
| meta-llama/Llama-3.1-8B-Instruct   | unsloth/Meta-Llama-3.1-8B-Instruct |   0.532 |   2.253 |  1.721 |
| HuggingFaceTB/SmolLM-135M-Instruct | unsloth/SmolLM-135M-Instruct       |   1.689 |   3.801 |  2.111 |
| Qwen/Qwen2.5-Coder-32B-Instruct    | unsloth/Qwen2.5-Coder-32B-Instruct |   0.220 |   3.081 |  2.861 |
