# P5 round 3: family x period effects and finer age bins (quasi-differenced GMM, real data)

| y            |   period | start        | groups   | modality   | spec                      |    est |    se |      z | converged   |   hansen_J |   J_df |   n_nodes |   n_groups |   n_periods |
|:-------------|---------:|:-------------|:---------|:-----------|:--------------------------|-------:|------:|-------:|:------------|-----------:|-------:|----------:|-----------:|------------:|
| first_noself |        7 | T0           | modality | all        | a0_7+a7_30+a30_90+a90_180 |  0.008 | 0.034 |  0.221 | True        |    123.378 |     20 |     15450 |          7 |         176 |
| first_noself |        7 | T0           | fam100   | all        | a0_7+a7_30+a30_90+a90_180 | -0.078 | 0.054 | -1.437 | True        |     84.929 |     20 |     15450 |        107 |         176 |
| first_noself |        7 | T0           | fam300   | all        | a0_7+a7_30+a30_90+a90_180 | -0.261 | 0.077 | -3.375 | True        |    195.028 |     20 |     15450 |        307 |         176 |
| first_noself |        7 | T0           | fam1000  | all        | a0_7+a7_30+a30_90+a90_180 | -0.958 | 0.172 | -5.560 | True        |    856.924 |     20 |     15450 |       1007 |         176 |
| first_noself |       14 | T0           | fam100   | all        | a0_7+a7_30+a30_90+a90_180 | -0.174 | 0.042 | -4.135 | True        |    153.489 |     20 |     15450 |        107 |          88 |
| first_noself |        7 | robust_start | fam100   | all        | a0_7+a7_30+a30_90+a90_180 |  0.048 | 0.052 |  0.924 | True        |     98.335 |     20 |     15394 |        107 |         132 |
| first        |        7 | T0           | fam100   | all        | a0_7+a7_30+a30_90+a90_180 | -0.093 | 0.059 | -1.575 | True        |    149.116 |     20 |     15450 |        107 |         176 |
| first_noself |        7 | T0           | fam100   | all        | base                      | -0.201 | 0.052 | -3.830 | True        |     63.255 |      8 |     15450 |        107 |         176 |
| first_noself |        7 | T0           | fam100   | text       | a0_7+a7_30+a30_90+a90_180 | -0.060 | 0.052 | -1.158 | True        |     80.956 |     20 |     11791 |        101 |         176 |
| first_noself |        7 | T0           | fam100   | image_gen  | a0_7+a7_30+a30_90+a90_180 | -0.400 | 0.295 | -1.359 | True        |    110.171 |     20 |       837 |        101 |         176 |

## Hansen J by family (main family spec, top 20)

|    root | family                                   |   J_f |   n_nodes |      y |
|--------:|:-----------------------------------------|------:|----------:|-------:|
|  916830 | Qwen/Qwen2.5-1.5B                        |   1.2 |        80 | 2142.0 |
|  916827 | Qwen/Qwen2.5-0.5B                        |   1.0 |        81 | 2757.0 |
| 1014273 | mistralai/Ministral-8B-Instruct-2410     |   0.6 |        16 |   21.0 |
|  733090 | meta-llama/Llama-3.1-8B                  |   0.6 |       242 | 2800.0 |
|  315616 | mistralai/Mistral-7B-v0.1                |   0.6 |       410 | 2894.0 |
| 1727007 | RekaAI/reka-flash-3.1                    |   0.6 |         1 |    1.0 |
| 1609822 | Qwen/Qwen3-0.6B-Base                     |   0.5 |        74 |  143.0 |
|  358375 | MAGAer13/mplug-owl2-llama2-7b            |   0.4 |         1 |    1.0 |
|  577684 | meta-llama/Meta-Llama-3-8B               |   0.4 |        80 | 2286.0 |
|  980624 | TroyDoesAI/BlackSheep-Llama3.2-3B        |   0.4 |         2 |    2.0 |
|  251764 | stabilityai/stable-diffusion-xl-base-1.0 |   0.4 |        65 | 2815.0 |
| 1188189 | deepseek-ai/DeepSeek-V3-Base             |   0.4 |         3 |   15.0 |
|   11078 | beomi/kcbert-base                        |   0.3 |         1 |  152.0 |
|  916835 | Qwen/Qwen2.5-7B                          |   0.3 |       169 | 1889.0 |
| 1387811 | google/gemma-3-4b-pt                     |   0.3 |        23 |  662.0 |
| 1727955 | black-forest-labs/FLUX.1-Kontext-dev     |   0.3 |         2 |   26.0 |
|  929495 | meta-llama/Llama-3.2-1B                  |   0.3 |        30 |  653.0 |
|  929543 | meta-llama/Llama-3.2-3B-Instruct         |   0.2 |        73 |  928.0 |
|  733062 | HuggingFaceTB/SmolLM-360M                |   0.2 |         5 |  201.0 |
| 1129598 | jaeunglee/resnet18-cifar10-unlearning    |   0.2 |         1 |    1.0 |
