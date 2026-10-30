# P12 mirror pairs: identical weights, different name / degree

44 pairs (copy declares the original as base model; same repo name; both in the snapshot).

## Estimator check on simulated pairs (mean of 20 seeds; sd of estimates in a_sd)

| data                    |      a |   a_sd |   a_se |
|:------------------------|-------:|-------:|-------:|
| simulated, true a = 0.0 | -0.002 |  0.022 |  0.021 |
| simulated, true a = 0.5 |  0.491 |  0.024 |  0.040 |
| simulated, true a = 1.0 |  0.970 |  0.084 |  0.089 |

## Real pairs: pair-FE binomial logit of the copy's share of new children on the log degree ratio

| outcome      | sample                    |      a |   a_se |   b_age |   b_age_se |   n_pair_months |   n_pairs |
|:-------------|:--------------------------|-------:|-------:|--------:|-----------:|----------------:|----------:|
| all          | all months                | -0.123 |  0.137 |   1.057 |      0.404 |             309 |        44 |
| all          | first 12 months           | -0.123 |  0.137 |   1.057 |      0.404 |             309 |        44 |
| all          | pairs with gap <= 30 days | -0.109 |  0.144 |   1.110 |      0.719 |             264 |        36 |
| first_noself | all months                | -0.101 |  0.160 |   1.331 |      0.715 |             308 |        44 |
| first_noself | first 12 months           | -0.101 |  0.160 |   1.331 |      0.715 |             308 |        44 |
| first_noself | pairs with gap <= 30 days | -0.047 |  0.171 |   2.803 |      1.275 |             264 |        36 |

## Pairs

| orig_id                                   | copy_id                               |   gap_days |   orig_deg_at_copy |   children_orig_after |   children_copy |   copy_share_after |
|:------------------------------------------|:--------------------------------------|-----------:|-------------------:|----------------------:|----------------:|-------------------:|
| Qwen/Qwen2.5-0.5B-Instruct                | unsloth/Qwen2.5-0.5B-Instruct         |       2.38 |                  0 |                   651 |             747 |               0.53 |
| meta-llama/Llama-3.2-3B-Instruct          | unsloth/Llama-3.2-3B-Instruct         |       7.14 |                  1 |                   843 |             582 |               0.41 |
| meta-llama/Llama-3.2-1B-Instruct          | unsloth/Llama-3.2-1B-Instruct         |       7.14 |                  0 |                  1194 |             537 |               0.31 |
| meta-llama/Llama-3.2-3B                   | unsloth/Llama-3.2-3B                  |       7.14 |                  0 |                   377 |             433 |               0.53 |
| meta-llama/Llama-3.2-1B                   | unsloth/Llama-3.2-1B                  |       7.15 |                  0 |                   819 |             339 |               0.29 |
| Qwen/Qwen2.5-1.5B-Instruct                | unsloth/Qwen2.5-1.5B-Instruct         |       1.29 |                  0 |                  1303 |             328 |               0.20 |
| microsoft/Phi-3.5-mini-instruct           | unsloth/Phi-3.5-mini-instruct         |       3.98 |                  0 |                   716 |             327 |               0.31 |
| Qwen/Qwen2.5-3B                           | unsloth/Qwen2.5-3B                    |       3.38 |                  4 |                   506 |             324 |               0.39 |
| Qwen/Qwen2.5-3B-Instruct                  | unsloth/Qwen2.5-3B-Instruct           |       1.30 |                  0 |                   965 |             321 |               0.25 |
| Qwen/Qwen2.5-1.5B                         | unsloth/Qwen2.5-1.5B                  |       3.37 |                  6 |                   458 |             296 |               0.39 |
| Qwen/Qwen2.5-0.5B                         | unsloth/Qwen2.5-0.5B                  |       3.37 |                  4 |                   648 |             294 |               0.31 |
| Qwen/Qwen2.5-Math-1.5B-Instruct           | unsloth/Qwen2.5-Math-1.5B-Instruct    |       6.74 |                  0 |                    59 |             287 |               0.83 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct          | unsloth/Qwen2.5-Coder-1.5B-Instruct   |       4.88 |                  2 |                    69 |             276 |               0.80 |
| HuggingFaceTB/SmolLM2-1.7B                | unsloth/SmolLM2-1.7B                  |       0.85 |                  0 |                    39 |             256 |               0.87 |
| HuggingFaceTB/SmolLM-360M                 | unsloth/SmolLM-360M                   |      63.23 |                 26 |                    11 |             255 |               0.96 |
| Qwen/Qwen2.5-Math-1.5B                    | unsloth/Qwen2.5-Math-1.5B             |       6.73 |                  3 |                   124 |             254 |               0.67 |
| HuggingFaceTB/SmolLM-360M-Instruct        | unsloth/SmolLM-360M-Instruct          |      62.28 |                 15 |                     6 |             248 |               0.98 |
| Qwen/Qwen2.5-Coder-7B                     | unsloth/Qwen2.5-Coder-7B              |       6.75 |                  4 |                    50 |             246 |               0.83 |
| Qwen/Qwen2.5-Coder-7B-Instruct            | unsloth/Qwen2.5-Coder-7B-Instruct     |       5.72 |                  1 |                   435 |             239 |               0.35 |
| Qwen/QwQ-32B                              | unsloth/QwQ-32B                       |       0.28 |                  1 |                    92 |             239 |               0.72 |
| Qwen/Qwen2.5-Math-7B-Instruct             | unsloth/Qwen2.5-Math-7B-Instruct      |       3.73 |                  0 |                   402 |             235 |               0.37 |
| HuggingFaceTB/SmolLM-1.7B-Instruct        | unsloth/SmolLM-1.7B-Instruct          |      62.18 |                 15 |                     8 |             234 |               0.97 |
| HuggingFaceTB/SmolLM2-360M                | unsloth/SmolLM2-360M                  |       0.86 |                  0 |                    66 |             218 |               0.77 |
| Qwen/Qwen2.5-14B-Instruct                 | unsloth/Qwen2.5-14B-Instruct          |       6.72 |                  6 |                   341 |             217 |               0.39 |
| HuggingFaceTB/SmolLM2-135M                | unsloth/SmolLM2-135M                  |       0.86 |                  0 |                   604 |             205 |               0.25 |
| google/gemma-3-1b-it                      | unsloth/gemma-3-1b-it                 |       1.79 |                  0 |                   293 |             193 |               0.40 |
| Qwen/Qwen2.5-7B-Instruct                  | unsloth/Qwen2.5-7B-Instruct           |       2.39 |                  0 |                  2773 |             161 |               0.05 |
| Qwen/Qwen3-8B                             | unsloth/Qwen3-8B                      |       1.01 |                  0 |                   157 |             161 |               0.51 |
| google/gemma-3-4b-it                      | unsloth/gemma-3-4b-it                 |      19.47 |                  0 |                   252 |             103 |               0.29 |
| Qwen/Qwen3-0.6B-Base                      | unsloth/Qwen3-0.6B-Base               |       0.31 |                  0 |                   298 |             100 |               0.25 |
| Qwen/Qwen3-4B                             | unsloth/Qwen3-4B                      |       1.17 |                  0 |                   146 |              88 |               0.38 |
| Qwen/Qwen2-VL-2B-Instruct                 | unsloth/Qwen2-VL-2B-Instruct          |      83.96 |                 46 |                   240 |              65 |               0.21 |
| meta-llama/Llama-3.2-11B-Vision-Instruct  | unsloth/Llama-3.2-11B-Vision-Instruct |       7.12 |                  0 |                   414 |              56 |               0.12 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | unsloth/DeepSeek-R1-Distill-Qwen-1.5B |       0.13 |                  0 |                   448 |              50 |               0.10 |
| google/gemma-3-27b-it                     | unsloth/gemma-3-27b-it                |      10.49 |                  0 |                   564 |              43 |               0.07 |
| Qwen/Qwen2.5-7B-Instruct                  | Gensyn/Qwen2.5-7B-Instruct            |     226.02 |               2156 |                   684 |              42 |               0.06 |
| Qwen/Qwen2.5-VL-3B-Instruct               | unsloth/Qwen2.5-VL-3B-Instruct        |       4.98 |                  8 |                   286 |              41 |               0.13 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | unsloth/DeepSeek-R1-Distill-Qwen-32B  |       0.23 |                  0 |                   242 |              41 |               0.14 |
| Qwen/Qwen2.5-7B                           | unsloth/Qwen2.5-7B                    |       3.38 |                  7 |                   912 |              41 |               0.04 |
| meta-llama/Llama-3.1-8B-Instruct          | unsloth/Llama-3.1-8B-Instruct         |     212.06 |               1534 |                   781 |              39 |               0.05 |
| Qwen/Qwen2.5-32B-Instruct                 | unsloth/Qwen2.5-32B-Instruct          |       6.04 |                  3 |                   250 |              23 |               0.08 |
| Qwen/Qwen2.5-VL-7B-Instruct               | unsloth/Qwen2.5-VL-7B-Instruct        |       5.04 |                  5 |                   428 |              21 |               0.05 |
| Qwen/Qwen2-VL-7B-Instruct                 | unsloth/Qwen2-VL-7B-Instruct          |      83.96 |                 93 |                   398 |              21 |               0.05 |
| meta-llama/Llama-3.1-8B                   | unsloth/Llama-3.1-8B                  |     215.51 |               1092 |                   666 |               6 |               0.01 |
