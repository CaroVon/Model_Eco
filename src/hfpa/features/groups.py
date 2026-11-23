"""v2 feature groups (four players) and shared definitions.

  pop      popularity in the broad sense: social signal (log_deg, lrec30, major_org), visibility / life cycle
           (has_deg, legacy, log_age; missing-information dummies size_na, txt_na, bench_na: whether a candidate has a
           size, a card, a benchmark is visibility, not demand), local search (lco2, ds_couse) and availability
           (lquant, tool formats, re-distributor accounts)
  qual     general quality: bench_z (model-index leaderboard scores); hub FE only in the upper-bound variants
  match    demand match: size_dist, task_hist, lang_hist, ds_lang, ds_task, txt_sim
  inertia  author_reuse, author_family, same_author, is_last
"""
import numpy as np
import pandas as pd

LIB = ["transformers", "peft", "diffusers", "sentence-transformers"]  # gguf / mlx / timm: separation (P15)
GROUPS = {
    "pop": ["log_deg", "lrec30", "major_org", "has_deg", "legacy", "log_age", "size_na", "txt_na", "bench_na",
            "lco2", "ds_couse", "lquant"] + [f"lib_{x}" for x in LIB] + ["redistributor"],
    "qual": ["bench_z"],
    "match": ["size_dist", "task_hist", "lang_hist", "ds_lang", "ds_task", "txt_sim"],
    "inertia": ["author_reuse", "author_family", "same_author", "is_last"],
}
ALL = sum(GROUPS.values(), [])
NAMES = {"pop": "popularity", "qual": "general quality", "match": "demand match", "inertia": "inertia",
         "self": "self-supply", "habit": "habit"}
# v2.1: the inertia group split into self-supply (the candidate is the author's own model: an organisational /
# vertical-integration choice) and habit (reusing a parent or family the author used before, last parent).
GROUPS5 = {"pop": GROUPS["pop"], "qual": GROUPS["qual"], "match": GROUPS["match"], "self": ["same_author"],
           "habit": ["author_reuse", "author_family", "is_last"]}
# popularity columns that count the author's own earlier children on repeat decisions (scripts/v2_selfdeg.py)
SELFDEG = {"log_deg": "log_deg_ex", "has_deg": "has_deg_ex", "lrec30": "lrec30_ex"}
SIGNAL = ["hist_has", "ds_has", "txt_has"]  # the choice carries a demand signal
MIN_TOKENS = 5
MAIN = ["finetune", "adapter", "unspecified"]


def group_of(f):
    return next(k for k, v in GROUPS.items() if f in v)


def redistributors(nodes, E, min_copies=20, min_share=0.5):
    """Re-distributor accounts (unsloth, TheBloke, ...): organisations with >= min_copies models that are copies --
    the repo name was first used by another organisation, or a quantized child of another author's model -- and
    copies >= min_share of their models. Returns (org names, bool per node)."""
    org = nodes.model_id.str.split("/").str[0]
    repo = nodes.model_id.str.split("/").str[-1].str.lower()
    o = np.argsort(nodes.birth.to_numpy(), kind="stable")
    first_org = pd.Series(org.to_numpy()[o], index=repo.to_numpy()[o])
    first_org = first_org[~first_org.index.duplicated()]
    twin = org.to_numpy() != first_org.reindex(repo.to_numpy()).to_numpy()
    a = nodes.author_code.to_numpy()
    q = E[E.rel_type == "quantized"]
    q = q[a[q.child_idx.to_numpy()] != a[q.parent_idx.to_numpy()]]
    copy = twin.copy()
    copy[q.child_idx.to_numpy()] = True
    g = pd.DataFrame({"org": org.to_numpy(), "copy": copy}).groupby("org")["copy"].agg(["size", "sum"])
    sel = g.index[(g["sum"] >= min_copies) & (g["sum"] / g["size"] >= min_share)]
    return sorted(sel), np.isin(org.to_numpy(), sel)
