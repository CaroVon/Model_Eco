import numpy as np

from hfpa.features.text import clean, drop_templates, name_vocab, para_key, row_cosine, tfidf, tokens

CARD = """---
license: mit
base_model: Qwen/Qwen2.5-7B
---
# Medical QA

This model is a fine-tuned version of [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) for medical
question answering in hospitals.

```python
from transformers import AutoModel
```

| lr | 1e-4 |
|----|------|

医学问答
"""


def test_clean_removes_markup_and_code():
    p = clean(CARD)
    s = " ".join(p)
    assert "license" not in s and "AutoModel" not in s and "https" not in s and "1e-4" not in s
    assert "medical" in s.lower() and "医学问答" in s


def test_tokens_drop_names_and_digits():
    names = name_vocab(["Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B"])
    assert {"qwen", "meta", "llama"} <= names
    t = tokens(clean(CARD), names)
    assert "qwen" not in t and not any(any(c.isdigit() for c in w) for w in t)
    assert "medical" in t and "hospitals" in t and "医学" in t


def test_drop_templates_and_cosine():
    parent = ["This is the official release of a strong base model for many uses.", "Contact us."]
    child = ["This is the official release of a strong base model for many uses.", "Answers legal questions in German."]
    banned = {para_key(p) for p in parent} - {None}
    assert drop_templates(child, banned) == ["Answers legal questions in German."]
    docs = [["legal", "german"], ["legal", "german", "law"], ["image", "cats"], ["image", "dogs"], ["law"]]
    X, _ = tfidf(docs, min_df=1, max_df=1.0)
    c = row_cosine(X, np.array([0, 0]), np.array([1, 2]))
    assert c[0] > 0.5 and c[1] == 0


def test_hub_name_tokens_and_sizes_removed():
    names = name_vocab([], full_ids=["Qwen/Qwen2.5-Coder-7B-Instruct"])
    t = tokens(["Whisper tiny Spanish, a coder model for German law"], names)
    assert "coder" not in t and "tiny" not in t and "spanish" in t and "law" in t


def test_mask_names_keeps_prose():
    from hfpa.features.text import mask_names
    names = {"qwen", "llama", "instruct"}
    out = mask_names(["Fine-tuned Qwen2.5-7B-Instruct for Korean medical QA, see org/repo."], names)
    assert "Qwen" not in out and "org/repo" not in out and "7B" not in out
    assert "Korean medical QA" in out and "Fine-tuned" in out


def test_v2_names_keep_descriptive_words():
    from hfpa.features.text import name_vocab_v2
    v = name_vocab_v2(["hfl/chinese-roberta-wwm-ext", "meta-llama/Llama-3-8B", "AI-MO/math-shepherd"])
    assert "llama" in v and "meta" in v and "hfl" in v
    assert "chinese" not in v and "math" not in v and "roberta" not in v  # only the first repo token is a family


def test_minhash_near_duplicates():
    import numpy as np
    from hfpa.features.text import lsh_near_dup, minhash, shingle_hashes
    rng = np.random.default_rng(0)
    base = rng.integers(0, 10_000, 60)
    near = base.copy()
    near[30] = 99_999  # one word changed: Jaccard of 5-shingles ~ 0.85
    other = rng.integers(0, 10_000, 60)
    words = np.concatenate([base, near, other])
    off = np.array([0, 60, 120, 180])
    h, own = shingle_hashes(words, off)
    sig = minhash(h, own, 3)
    est = (sig[0] == sig[1]).mean()
    assert est > 0.7 and (sig[0] == sig[2]).mean() < 0.1
    fo, fn, cdf = lsh_near_dup(sig, birth=np.array([1.0, 5.0, 2.0]), node=np.array([10, 11, 12]), df=np.array([1, 1, 1]),
                               bands=21, rows_per_band=3, thresh=0.7)
    assert fo[1] == 1.0 and fn[1] == 10 and np.isinf(fo[2]) and cdf[0] == 2
