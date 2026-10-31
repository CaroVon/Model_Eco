# P14 benchmark coverage (model-index in model cards, native dataset)

Benchmarks reported by >= 200 models: 12; models with bench_z: 3,476

| set                     |       n |   with_bench |   share |
|:------------------------|--------:|-------------:|--------:|
| all nodes               | 1860413 |         3476 |   0.002 |
| parents (>= 1 child)    |   16269 |          360 |   0.022 |
| parents >= 100 children |     508 |            9 |   0.018 |

## Most reported benchmarks

| bench                                                                  |   models |
|:-----------------------------------------------------------------------|---------:|
| idavidrein/gpqa|acc_norm                                               |     1793 |
| taur-lab/musr|acc_norm                                                 |     1792 |
| tiger-lab/mmlu-pro|acc                                                 |     1792 |
| winogrande|acc                                                         |     1774 |
| cais/mmlu|acc                                                          |     1725 |
| gsm8k|acc                                                              |     1719 |
| hellaswag|acc_norm                                                     |     1707 |
| ai2_arc|acc_norm                                                       |     1707 |
| truthful_qa|mc2                                                        |     1707 |
| bbh|acc_norm                                                           |     1589 |
| huggingfaceh4/ifeval|inst_level_strict_acc and prompt_level_strict_acc |     1588 |
| hendrycks/competition_math|exact_match                                 |     1588 |
