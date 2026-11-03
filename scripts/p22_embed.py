"""P22: better demand signals from model cards (sentence embeddings) and an LLM-labelled validation set.

Stages (env in brackets; [qc-vllm] = conda env and models BORROWED read-only from the ICML project, see README):
  --prep        [hfpa]     kept paragraphs of P18 (same template rules), names / sizes / digits masked but prose
                           kept (hfpa.features.text.mask_names) -> interim/cards_masked.parquet
  --embed       [qc-vllm]  intfloat/multilingual-e5-base (downloaded to ACMEC external/models, not borrowed),
                           mean pooling, "query: " prefix on both sides -> interim/cards_emb.npy
  --build       [hfpa]     emb_sim = cosine of child and candidate embeddings (same missing rule as txt_sim);
                           processed/choices/<spec>/emb.parquet; P15 full model and Shapley shares with emb_sim
                           instead of / in addition to txt_sim
  --label-prep  [hfpa]     300 plan-A choices (150 with a demand signal, 150 without): child purpose (masked repo
                           name + masked own card text) vs two candidates in random order: the chosen parent and the
                           most popular other hub of the choice set (P15 popularity index). Prompts ->
                           outputs/p22/label_input.parquet; 50 of them -> results/tables/p22_audit.csv for the user
  --label       [qc-vllm]  vllm, Qwen2.5-14B-Instruct (main) and Llama-3.1-8B-Instruct (second rater), both from
                           /net/spaces/jevans/klab/caroyf/data/ICML/models/hf (borrowed) -> outputs/p22/labels_*.parquet
  --label-eval  [hfpa]     rater agreement; LLM verdict vs P15 per-choice type; share of choices whose chosen parent
                           fits better than the popular alternative, with and without a demand signal; agreement with
                           the user's audit once filled in
usage: sbatch slurm/cpu.sbatch scripts/p22_embed.py --prep ; sbatch slurm/gpu_vllm.sbatch scripts/p22_embed.py --embed
       sbatch slurm/gpu.sbatch scripts/p22_embed.py --build ; python scripts/p22_embed.py --label-prep
       sbatch slurm/gpu_vllm.sbatch scripts/p22_embed.py --label ; python scripts/p22_embed.py --label-eval
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from hfpa import paths

SPEC = "main_strat_frac0.1"
E5 = paths.EXTERNAL / "models" / "multilingual-e5-base"
ICML = "/net/spaces/jevans/klab/caroyf/data/ICML/models/hf"
RATERS = {"qwen14b": f"{ICML}/Qwen2.5-14B-Instruct", "llama8b": f"{ICML}/Llama-3.1-8B-Instruct"}
OUT = paths.OUTPUTS / "p22"
N_LABEL, N_AUDIT = 300, 50
BORROW = ("NOTE: borrowing read-only from the ICML project (user approval 2026-10-31): conda env qc-vllm and the "
          "local models in /net/spaces/jevans/klab/caroyf/data/ICML/models/hf; nothing in ICML is modified.")


# ---------------------------------------------------------------- embeddings
def prep():
    from hfpa.features.text import MIN_TOKENS, mask_names, tokens
    from p18_text import kept_paragraphs
    t0 = time.time()
    nodes, C, child_nodes, cand_nodes, kept, names = kept_paragraphs(t0)
    side = np.r_[np.zeros(len(child_nodes), int), np.ones(len(cand_nodes), int)]
    node = np.r_[child_nodes, cand_nodes]
    text = [mask_names(ps, names) for ps in kept]
    ok = np.array([len(tokens(ps, names)) >= MIN_TOKENS for ps in kept])
    pd.DataFrame({"node": node, "side": side, "text": text, "ok": ok}).to_parquet(paths.INTERIM / "cards_masked.parquet", index=False)
    print(f"{len(node):,} documents, {ok.sum():,} with >= {MIN_TOKENS} tokens; {time.time() - t0:.0f}s", flush=True)


def embed(bs=512):
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(BORROW + " (here: the qc-vllm env only; the embedding model is ACMEC's own download)", flush=True)
    T = pd.read_parquet(paths.INTERIM / "cards_masked.parquet")
    idx = np.flatnonzero(T.ok.to_numpy())
    texts = ["query: " + t for t in T.text.to_numpy()[idx]]
    tok = AutoTokenizer.from_pretrained(E5)
    model = AutoModel.from_pretrained(E5, torch_dtype=torch.float16).cuda().eval()
    order = np.argsort([len(t) for t in texts])  # length-sorted batches
    out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), bs):
            o = order[s:s + bs]
            enc = tok([texts[i] for i in o], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            out[o] = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy().astype(np.float16)
            if s // bs % 200 == 0:
                print(f"{s:,}/{len(order):,} in {time.time() - t0:.0f}s", flush=True)
    np.save(paths.INTERIM / "cards_emb.npy", out)
    np.save(paths.INTERIM / "cards_emb_idx.npy", idx)
    print(f"done {len(idx):,} in {time.time() - t0:.0f}s", flush=True)


def emb_sims(kid, cand):
    T = pd.read_parquet(paths.INTERIM / "cards_masked.parquet", columns=["node", "side"])
    E = np.load(paths.INTERIM / "cards_emb.npy")
    idx = np.load(paths.INTERIM / "cards_emb_idx.npy")
    n = int(max(T.node.max(), kid.max(), cand.max())) + 1
    erow = np.full(len(T), -1)
    erow[idx] = np.arange(len(idx))
    crow, prow = np.full(n, -1), np.full(n, -1)
    c, p = T.side.to_numpy() == 0, T.side.to_numpy() == 1
    crow[T.node.to_numpy()[c]] = erow[c]
    prow[T.node.to_numpy()[p]] = erow[p]
    a, b = crow[kid], prow[cand]
    ok = (a >= 0) & (b >= 0)
    s = np.zeros(len(kid), np.float32)
    for st in range(0, ok.sum(), 2_000_000):
        sel = np.flatnonzero(ok)[st:st + 2_000_000]
        s[sel] = (E[a[sel]].astype(np.float32) * E[b[sel]].astype(np.float32)).sum(1)
    return s, (~ok).astype(np.float32)


def build():
    import p15_decomp as P15
    from p19_robust import prep as p19prep, shares
    rows, ch = P15.load()
    d = paths.PROCESSED / "choices" / SPEC
    kid = ch.child.loc[rows.choice_id].to_numpy()
    s, na = emb_sims(kid, rows.cand_id.to_numpy())
    pd.DataFrame({"choice_id": rows.choice_id.to_numpy(), "cand_id": rows.cand_id.to_numpy(), "emb_sim": s, "emb_na": na}).to_parquet(d / "emb.parquet", index=False)
    rows["emb_sim"], rows["emb_na"] = s, na
    # description (plan-A first choices): chosen parent's percentile among candidates with both texts
    r, _ = P15.sample(rows, ch, "planA")
    desc = []
    for c in ("txt_sim", "emb_sim"):
        ok = r[r[c.replace("sim", "na")] == 0]
        rk = ok.groupby("choice_id")[c].rank(ascending=False, method="average")
        n = ok.groupby("choice_id")[c].transform("size")
        pr = (1 - (rk - 1) / n.clip(lower=1))[ok.chosen == 1]
        desc.append(dict(measure=c, rows_with_both=len(ok) / len(r), chosen_pct_rank=pr.mean(),
                         sim_chosen=ok.loc[ok.chosen == 1, c].mean(), sim_other=ok.loc[ok.chosen == 0, c].mean()))
    out, sh = [], []
    match0 = list(P15.GROUPS["match"])
    for tag, match in [("tf-idf (P18)", match0), ("embedding instead", [f.replace("txt_", "emb_") for f in match0]),
                       ("both", match0 + ["emb_sim"])]:
        P15.GROUPS["match"][:] = match
        tr, cid, w = p19prep(rows, ch)
        feats = P15.drop_constant(tr, sum(P15.GROUPS.values(), []))
        res = P15.fit(tr, ch, feats, 0.0, cluster=True)
        for i, f in enumerate(feats):
            if f in ("txt_sim", "emb_sim", "txt_na", "emb_na"):
                out.append(dict(model=tag, param=f, coef=res["coef"][i], se_cl=res["se_cl"][i], ll=res["ll"]))
        sh.append(shares(tr, w).assign(model=tag))
        print(tag, out[-1], flush=True)
    P15.GROUPS["match"][:] = match0
    P15.P5.write(pd.DataFrame(desc), "p22_emb_desc", "P22 card similarity of the chosen parent vs other candidates, TF-IDF vs embedding (plan A)")
    P15.P5.write(pd.DataFrame(out), "p22_emb_coef", "P22 text-similarity coefficients in the P15 full model (plan A, clustered SE)")
    P15.P5.write(pd.concat(sh, ignore_index=True), "p22_emb_shapley", "P22 Shapley shares with the TF-IDF / embedding text measure")


# ---------------------------------------------------------------- LLM labels
PROMPT = """You are helping to evaluate how developers choose base models on Hugging Face.
A developer published a fine-tuned model. Its purpose, as far as it can be told (model names and sizes are masked):
  repository name: {child_name}
  task tag: {child_task}
  description: {child_text}

The developer could build it on either of these two base models:
[A] {a_id}  (task tag: {a_task}; parameters: {a_params})
    card excerpt: {a_text}
[B] {b_id}  (task tag: {b_task}; parameters: {b_params})
    card excerpt: {b_text}

Which base model is the better fit for this purpose (task, language, domain, modality, size)? Ignore which one is
more popular or better known. Answer with JSON only:
{{"purpose": "<at most 12 words>", "better": "A" | "B" | "equal" | "unclear", "confidence": 1-5}}
"equal" = both fit about equally well; "unclear" = the purpose cannot be told from the information given."""


def label_prep():
    import p15_decomp as P15
    from hfpa.features.text import mask_names
    from p18_text import kept_paragraphs
    rows, ch = P15.load()
    r, g = P15.sample(rows, ch, "planA")
    C = pd.read_csv(paths.TABLES / "p15_clogit.csv")
    b = C[C.model == "planA"].set_index("param").coef
    r = r.assign(pop=sum(b.get(f, 0) * r[f] for f in P15.GROUPS["pop"]))
    hub = r[(r.stratum == P15.P9.HUB) & (r.chosen == 0)]
    alt = hub.loc[hub.groupby("choice_id")["pop"].idxmax(), ["choice_id", "cand_id"]].set_index("choice_id").cand_id
    K = pd.read_parquet(paths.PROCESSED / "choices" / SPEC / "p15_classify.parquet")
    cid = np.intersect1d(K.index, alt.index)
    K = K.loc[cid]
    rng = np.random.default_rng(22)
    sig = K.index[K["type"] != "no demand signal"].to_numpy()
    nos = K.index[K["type"] == "no demand signal"].to_numpy()
    pick = np.r_[rng.choice(sig, N_LABEL // 2, replace=False), rng.choice(nos, N_LABEL // 2, replace=False)]
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id", "pipeline_tag", "params_b"])
    T = pd.read_parquet(paths.INTERIM / "cards_masked.parquet")
    own = dict(zip(T.node[T.side == 0], T.text[T.side == 0]))
    cards = pd.read_parquet(paths.INTERIM / "cards_clean.parquet").set_index("node").paras
    _, _, _, _, _, names = kept_paragraphs()
    recs = []
    for k in pick:
        c, p, a = int(ch.child.at[k]), int(ch.parent.at[k]), int(alt.at[k])
        flip = rng.random() < 0.5
        A, B = (a, p) if flip else (p, a)
        info = lambda v: dict(id=nodes.model_id.iat[v], task=nodes.pipeline_tag.iat[v] or "none",
                              params=f"{nodes.params_b.iat[v]:.2g}B" if np.isfinite(nodes.params_b.iat[v]) else "unknown",
                              text=" ".join(cards.get(v, []))[:700] or "(no card)")
        ia, ib = info(A), info(B)
        cname = mask_names([nodes.model_id.iat[c].split("/")[-1].replace("-", " ").replace("_", " ")], names) or "(masked)"
        prompt = PROMPT.format(child_name=cname, child_task=nodes.pipeline_tag.iat[c] or "none",
                               child_text=own.get(c, "")[:1200] or "(none)",
                               a_id=ia["id"], a_task=ia["task"], a_params=ia["params"], a_text=ia["text"],
                               b_id=ib["id"], b_task=ib["task"], b_params=ib["params"], b_text=ib["text"])
        recs.append(dict(choice_id=k, child=nodes.model_id.iat[c], chosen=nodes.model_id.iat[p], popular_alt=nodes.model_id.iat[a],
                         chosen_is=("B" if flip else "A"), p15_type=K.at[k, "type"], prompt=prompt))
    L = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_parquet(OUT / "label_input.parquet", index=False)
    au = L.sample(N_AUDIT, random_state=0)[["choice_id", "prompt"]].assign(your_answer_A_B_equal_unclear="")
    au.to_csv(paths.TABLES / "p22_audit.csv", index=False)
    print(L.p15_type.value_counts(), flush=True)
    print(L.prompt.iloc[0])


def label():
    from vllm import LLM, SamplingParams
    print(BORROW, flush=True)
    L = pd.read_parquet(OUT / "label_input.parquet")
    for name, path in RATERS.items():
        llm = LLM(model=path, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, seed=0)
        msgs = [[{"role": "user", "content": p}] for p in L.prompt]
        res = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=120))
        txt = [o.outputs[0].text for o in res]
        L[f"raw_{name}"] = txt
        L[f"better_{name}"] = [parse(t).get("better", "parse_error") for t in txt]
        L[f"purpose_{name}"] = [parse(t).get("purpose", "") for t in txt]
        L[f"conf_{name}"] = [parse(t).get("confidence", np.nan) for t in txt]
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print(name, L[f"better_{name}"].value_counts().to_dict(), flush=True)
    L.drop(columns="prompt").to_parquet(OUT / "labels.parquet", index=False)


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


def verdict(better, chosen_is):
    if better in ("A", "B"):
        return "chosen fits better" if better == chosen_is else "popular alternative fits better"
    return {"equal": "equal", "unclear": "unclear"}.get(better, "parse error")


def label_eval():
    import p15_decomp as P15
    L = pd.read_parquet(OUT / "labels.parquet")
    for n in RATERS:
        L[f"v_{n}"] = [verdict(b, c) for b, c in zip(L[f"better_{n}"], L.chosen_is)]
    L["signal"] = np.where(L.p15_type == "no demand signal", "no demand signal", "with demand signal")
    out = []
    for (grp, val), sub in [(("all", ""), L)] + [(("signal", k), s) for k, s in L.groupby("signal")] + \
                           [(("p15 type", k), s) for k, s in L.groupby("p15_type")]:
        for n in RATERS:
            vc = sub[f"v_{n}"].value_counts(normalize=True)
            out.append(dict(group=grp, value=val, rater=n, n=len(sub), **{k: vc.get(k, 0.0) for k in
                       ["chosen fits better", "equal", "popular alternative fits better", "unclear", "parse error"]}))
    agree = (L.v_qwen14b == L.v_llama8b).mean()
    det = L[~L.v_qwen14b.isin(["unclear", "parse error"]) & ~L.v_llama8b.isin(["unclear", "parse error"])]
    out.append(dict(group="rater agreement", value="qwen14b vs llama8b", rater="both", n=len(L), agreement=agree,
                    agreement_both_determinate=(det.v_qwen14b == det.v_llama8b).mean(), n_both_determinate=len(det)))
    au = pd.read_csv(paths.TABLES / "p22_audit.csv")
    col = "your_answer_A_B_equal_unclear"
    if au[col].notna().any() and (au[col].astype(str).str.strip() != "").any():
        au = au[au[col].astype(str).str.strip().str.len() > 0].merge(L, on="choice_id")
        au["v_user"] = [verdict(str(b).strip().upper() if str(b).strip().upper() in "AB" else str(b).strip().lower(), c)
                        for b, c in zip(au[col], au.chosen_is)]
        for n in RATERS:
            out.append(dict(group="agreement with user audit", value="", rater=n, n=len(au),
                            agreement=(au.v_user == au[f"v_{n}"]).mean()))
    P15.P5.write(pd.DataFrame(out), "p22_labels", "P22 LLM verdicts: chosen parent vs the most popular other hub, by demand signal and P15 type")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for s in ("prep", "embed", "build", "label-prep", "label", "label-eval"):
        ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    for s in ("prep", "embed", "build", "label_prep", "label", "label_eval"):
        if getattr(a, s):
            globals()[s]()
