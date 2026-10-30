# Data audit (P1, checkpoint 1)

Source: `modelbiome/ai_ecosystem_withmodelcards`, file `ai_ecosystem_withmodelcards.csv` (5,194,246,959 bytes); parsed without `modelCard`.

## 1. Rows and ids
- rows: 1,860,411
- `model_id` unique: True
- `private` values: {'False': 1860411}
- no `author` column in the dataset: author = `model_id` prefix before `/`
- rows with `exception_raised` set: 619,197 (scrape issues with model cards; metadata still present)

## 2. createdAt
- raw format example: `2025-07-11T00:55:12.000Z` (ISO 8601, UTC `Z`)
- unparseable: 0
- min: 2022-03-02 23:29:04+00:00  max (snapshot date): 2025-07-12 17:14:09+00:00
- most frequent timestamps:
  - 2022-03-02 23:29:05+00:00: 22,036
  - 2022-03-02 23:29:04+00:00: 7,758
  - 2025-03-20 03:50:32+00:00: 85
  - 2024-12-20 21:56:16+00:00: 83
  - 2022-06-12 06:00:15+00:00: 60
- backfilled block = createdAt <= 2022-03-02 23:29:05+00:00: 29,794 models (the two timestamps 23:29:04 and 23:29:05 on 2022-03-02); these form the initial graph G0 (legacy=1).
- study window: choices with child createdAt in (2022-03-02 23:29:05+00:00, 2025-07-12 17:14:09+00:00]

## 3. Parent tags (20 random models with a base_model tag)
- `minhnguyennnnnn/e828623a-4dd0-4be6-a733-770c858c2cb9`: `base_model:NousResearch/Nous-Hermes-2-Mistral-7B-DPO`, `base_model:adapter:NousResearch/Nous-Hermes-2-Mistral-7B-DPO`
- `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF`: `base_model:Qwen/Qwen2.5-Coder-32B-Instruct`, `base_model:quantized:Qwen/Qwen2.5-Coder-32B-Instruct`
- `SangwooJ/marian-finetuned-kde4-en-to-fr`: `base_model:Helsinki-NLP/opus-mt-en-fr`, `base_model:finetune:Helsinki-NLP/opus-mt-en-fr`
- `xueyj/Qwen-Qwen1.5-0.5B-1725933317`: `base_model:Qwen/Qwen1.5-0.5B`, `base_model:adapter:Qwen/Qwen1.5-0.5B`
- `glif-loradex-trainer/insectagon_Moths_opal_fire`: `base_model:black-forest-labs/FLUX.1-dev`, `base_model:finetune:black-forest-labs/FLUX.1-dev`, `base_model:adapter:black-forest-labs/FLUX.1-dev`
- `VidyutCx/mistral-prompttune`: `base_model:mistralai/Mistral-7B-Instruct-v0.3`, `base_model:finetune:mistralai/Mistral-7B-Instruct-v0.3`
- `searchone/boo-king-blue`: `base_model:black-forest-labs/FLUX.1-dev`, `base_model:adapter:black-forest-labs/FLUX.1-dev`
- `bbytxt/850397bc-22b6-475c-9d32-6e334db7b2f1`: `base_model:fxmarty/really-tiny-falcon-testing`, `base_model:adapter:fxmarty/really-tiny-falcon-testing`
- `Krabat/Qwen-Qwen1.5-0.5B-1723250479`: `base_model:Qwen/Qwen1.5-0.5B`, `base_model:adapter:Qwen/Qwen1.5-0.5B`
- `Aakali/llama-2-7b-chat-bird_only`: `base_model:codellama/CodeLlama-7b-hf`, `base_model:adapter:codellama/CodeLlama-7b-hf`
- `GeorgyGUF/adidas-tracksuit-with-three-stripes-lora-experiments`: `base_model:black-forest-labs/FLUX.1-dev`, `base_model:adapter:black-forest-labs/FLUX.1-dev`
- `dllmchh/Qwen-Qwen1.5-1.8B-1727692842`: `base_model:Qwen/Qwen1.5-1.8B`, `base_model:adapter:Qwen/Qwen1.5-1.8B`
- `mradermacher/StruXGPT-v2-Llama2-7B-GGUF`: `base_model:StruXGPT/StruXGPT-v2-Llama2-7B`, `base_model:quantized:StruXGPT/StruXGPT-v2-Llama2-7B`
- `BootesVoid/cmb20604o060au1cgqxta8xc3_cmb2fdxyv06bgu1cg4f8qrdn6`: `base_model:black-forest-labs/FLUX.1-dev`, `base_model:adapter:black-forest-labs/FLUX.1-dev`
- `shopitalic/Chloe-Teddy-Bombers-Winter-Style`: `base_model:black-forest-labs/FLUX.1-dev`, `base_model:adapter:black-forest-labs/FLUX.1-dev`
- `kkokas/Qwen-Qwen1.5-0.5B-1724132291`: `base_model:Qwen/Qwen1.5-0.5B`, `base_model:adapter:Qwen/Qwen1.5-0.5B`
- `unsloth/KernelLLM-GGUF`: `base_model:facebook/KernelLLM`, `base_model:quantized:facebook/KernelLLM`
- `MohammadPourbahram/whisper-small-fa`: `base_model:openai/whisper-small`, `base_model:finetune:openai/whisper-small`
- `wufengze/dreambooth-sd3-lora`: `base_model:stabilityai/stable-diffusion-3-medium-diffusers`, `base_model:adapter:stabilityai/stable-diffusion-3-medium-diffusers`
- `HeAAAAA/Crab`: `base_model:meta-llama/Llama-3.1-8B`, `base_model:finetune:meta-llama/Llama-3.1-8B`

Tag format confirmed: HF writes both a bare `base_model:<id>` and a typed `base_model:<rel>:<id>` tag; the parser keeps the typed one. Relation counts from tags: adapter=244,853, finetune=191,669, quantized=104,205, merge=41,039, unspecified=832.
- comparison with the dataset's own `*_parent` columns: {'both': 582960, 'right_only': 832, 'left_only': 0}; the tag-only rows are untyped (`unspecified`) parents, treated as finetune in the main spec.

## 4. Snapshot
- snapshot (max createdAt): 2025-07-12 17:14:09+00:00

## pipeline_tag
- missing: 1,146,564 (61.6%)
- tags not in modality map: []

# P2: edges and nodes

## Cleaning funnel
| step                                                    |   kept |   removed |
|:--------------------------------------------------------|-------:|----------:|
| parsed base_model tags                                  | 582598 |         0 |
| dedupe (child, parent) pairs                            | 582001 |       597 |
| parent matched exactly                                  | 573840 |         0 |
| parent matched case-insensitively                       |     25 |         0 |
| drop self loops                                         | 580736 |      1265 |
| dangling parents kept as placeholders (>= 100 children) |   3582 |         0 |
| drop dangling parents (not in snapshot)                 | 576182 |      4554 |
| drop child created before parent                        | 574531 |      1651 |
| drop equal timestamps (non-legacy)                      | 574531 |         0 |
| drop edges on cycles                                    | 574531 |         0 |

- dangling share before removal: 1.40% (< 2% -> removed, plan 5.7); largest:
  - `runwayml/stable-diffusion-v1-5`: 3,192 child edges
  - `togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference`: 390 child edges
  - `hZzy/qwen2.5-0.5b-sft-news-IFT`: 74 child edges
  - `WizardLM/WizardMath-7B-V1.1`: 39 child edges
  - `Dynosaur/llama3-8b-math-sft`: 35 child edges
  - `togethercomputer/Meta-Llama-3.1-8B-Instruct-Reference__TOG__FT`: 35 child edges
  - `AnonymousCS/populism_multilingual_roberta_base`: 31 child edges
  - `c14kevincardenas/ClimBEiT-t2`: 31 child edges
  - `Sao10K/L3-8B-Stheno-v3.1`: 30 child edges
  - `c14kevincardenas/ClimBEiT-t3`: 28 child edges
- placeholder nodes (>= 100 children, kept with flag `placeholder`): `runwayml/stable-diffusion-v1-5` (birth 2022-10-20, tag text-to-image), `togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference` (birth 2025-01-21, tag nan)

## Edges after cleaning
| rel_type    |   count |
|:------------|--------:|
| adapter     |  243643 |
| finetune    |  188819 |
| quantized   |  101095 |
| merge       |   40220 |
| unspecified |     754 |

## Top-10 parents (main spec: finetune+adapter+unspecified)
| parent                                   |   count |
|:-----------------------------------------|--------:|
| black-forest-labs/FLUX.1-dev             |   32984 |
| Qwen/Qwen1.5-0.5B                        |   32476 |
| Qwen/Qwen1.5-1.8B                        |   30572 |
| google/gemma-2b                          |   23761 |
| google/gemma-7b                          |    9543 |
| distilbert/distilbert-base-uncased       |    9329 |
| stabilityai/stable-diffusion-xl-base-1.0 |    8417 |
| Qwen/Qwen1.5-7B                          |    6434 |
| google-bert/bert-base-uncased            |    5542 |
| aubmindlab/bert-base-arabertv02          |    4000 |

## Nodes
- nodes: 1,860,413; legacy: 29,794; roots (no parent): 1,313,508
- modality: {'unknown': 840948, 'text': 764428, 'image_gen': 94373, 'other': 63638, 'speech': 40873, 'vision': 28973, 'multimodal': 27180}
- modality source: {'lib/unknown': 839778, 'own': 713848, 'parent': 306787}
- generation: {0: 1313508, 1: 418185, 2: 84525, 3: 32131, 4: 7757, 5: 1914, 6: 883, 7: 367}
- major_org models: 8,454; params parsed: 504,159
