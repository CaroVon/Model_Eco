"""General quality from the native dataset: benchmark results in model cards' model-index (YAML front matter),
and a decomposition of the hub fixed effects into a benchmark-explained part and a remainder.

--extract   read model_id + modelCard from the raw CSV, parse the model-index block of every card that has one,
            write interim/model_index.parquet (model_id, task, dataset, metric, value)
--analyze   node-level quality bench_z: each Open LLM Leaderboard benchmark (v1: ARC, HellaSwag, MMLU, TruthfulQA,
            Winogrande, GSM8k; v2: IFEval, BBH, MATH, GPQA, MuSR, MMLU-Pro) standardized across models, averaged
            over the benchmarks a model reports.
            Hub FE phi_j (p9_mixture2_g0.5_fe.npz, popularity mode) regressed on bench_z, log params, major org,
            modality: share of var(phi) by group; the remainder cannot be attributed and is assigned to
            popularity (lower bound for fit) or fit (upper bound) in the ratio stage (P15).
Writes results/tables/p14_bench_coverage.md, p14_phi_decomp.md/.csv and processed/nodes/bench.parquet.

usage: python scripts/p14_quality.py --extract   (Slurm: reads the 5 GB CSV)
       python scripts/p14_quality.py --analyze
"""
import argparse
import re

import numpy as np
import pandas as pd
import yaml

from hfpa import paths
from hfpa.data.load_meta import CSV

try:
    Loader = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    Loader = yaml.SafeLoader
FRONT = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*(\n|\Z)", re.S)
MIN_MODELS = 200
# Standardized, comparable benchmarks only (Open LLM Leaderboard v1 and v2, as written into model cards); most
# other model-index entries are evaluations on the developer's own data (imagefolder, glue, emotion, ...).
LEADERBOARD = {("ai2_arc", "acc_norm"), ("hellaswag", "acc_norm"), ("cais/mmlu", "acc"), ("truthful_qa", "mc2"),
               ("winogrande", "acc"), ("gsm8k", "acc"), ("huggingfaceh4/ifeval", "inst_level_strict_acc and prompt_level_strict_acc"),
               ("bbh", "acc_norm"), ("hendrycks/competition_math", "exact_match"), ("idavidrein/gpqa", "acc_norm"),
               ("taur-lab/musr", "acc_norm"), ("tiger-lab/mmlu-pro", "acc")}


def parse_card(card):
    m = FRONT.match(card)
    if not m:
        return []
    try:
        meta = yaml.load(m.group(1), Loader=Loader)
    except yaml.YAMLError:
        return []
    out = []
    mi = meta.get("model-index") if isinstance(meta, dict) else None
    if not isinstance(mi, list):
        return []
    for entry in mi:
        for r in (entry.get("results") or []) if isinstance(entry, dict) else []:
            if not isinstance(r, dict):
                continue
            task = (r.get("task") or {}).get("type") if isinstance(r.get("task"), dict) else None
            ds = r.get("dataset") or {}
            dname = ds.get("type") or ds.get("name") if isinstance(ds, dict) else None
            cfg = ds.get("config") if isinstance(ds, dict) else None
            for mt in r.get("metrics") or []:
                if isinstance(mt, dict):
                    try:
                        v = float(mt.get("value"))
                    except (TypeError, ValueError):
                        continue
                    out.append((task, str(dname), str(cfg), str(mt.get("type")), v))
    return out


def extract(chunksize=50_000):
    recs = []
    n = 0
    for ch in pd.read_csv(CSV, usecols=["model_id", "modelCard"], chunksize=chunksize, dtype=str,
                          keep_default_na=False, na_values=[""]):
        n += len(ch)
        sub = ch[ch.modelCard.fillna("").str.contains("model-index", regex=False)]
        for mid, card in zip(sub.model_id, sub.modelCard):
            for r in parse_card(card):
                recs.append((mid, *r))
        print(f"{n:,} cards read, {len(recs):,} results", flush=True)
    df = pd.DataFrame(recs, columns=["model_id", "task", "dataset", "config", "metric", "value"])
    df.to_parquet(paths.INTERIM / "model_index.parquet", index=False)
    print(df.groupby(["dataset", "metric"]).model_id.nunique().sort_values(ascending=False).head(40).to_string())


def analyze():
    mi = pd.read_parquet(paths.INTERIM / "model_index.parquet")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "modality", "params_b", "major_org"])
    idx = pd.Series(np.arange(len(nodes)), index=nodes.model_id)
    mi = mi[mi.model_id.isin(idx.index)].copy()
    key = list(zip(mi.dataset.str.lower(), mi.metric.str.lower()))
    mi = mi[[k in LEADERBOARD for k in key]].copy()
    mi["bench"] = mi.dataset.str.lower() + "|" + mi.metric.str.lower()
    mi = mi.groupby(["model_id", "bench"]).value.mean().reset_index()  # one value per model and benchmark
    cnt = mi.groupby("bench").model_id.nunique()
    keep = cnt.index[cnt >= MIN_MODELS]
    mi = mi[mi.bench.isin(keep)]
    mi["z"] = mi.groupby("bench").value.transform(lambda v: (v - v.mean()) / v.std())
    q = mi.groupby("model_id").agg(bench_z=("z", "mean"), n_bench=("z", "size")).reset_index()
    q["node"] = idx[q.model_id].to_numpy()
    q.to_parquet(paths.PROCESSED / "nodes" / "bench.parquet", index=False)

    # coverage: all nodes, nodes with >= 1 main child, hubs
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet", columns=["parent_idx", "rel_type"])
    deg = E[E.rel_type.isin(["finetune", "adapter", "unspecified"])].parent_idx.value_counts()
    has_b = np.zeros(len(nodes), bool)
    has_b[q.node] = True
    cov = []
    for label, sel in [("all nodes", np.ones(len(nodes), bool)),
                       ("parents (>= 1 child)", np.isin(np.arange(len(nodes)), deg.index)),
                       ("parents >= 100 children", np.isin(np.arange(len(nodes)), deg.index[deg >= 100]))]:
        cov.append(dict(set=label, n=int(sel.sum()), with_bench=int((sel & has_b).sum()), share=(sel & has_b).sum() / sel.sum()))
    cov = pd.DataFrame(cov)
    top = cnt[keep].sort_values(ascending=False).head(25).rename("models").reset_index()
    (paths.TABLES / "p14_bench_coverage.md").write_text("\n".join([
        "# P14 benchmark coverage (model-index in model cards, native dataset)", "",
        f"Benchmarks reported by >= {MIN_MODELS} models: {len(keep)}; models with bench_z: {len(q):,}", "",
        cov.to_markdown(index=False, floatfmt=".3f"), "", "## Most reported benchmarks", "",
        top.to_markdown(index=False)]) + "\n")

    # phi decomposition
    import statsmodels.api as sm
    z = np.load(paths.TABLES / "p9_mixture2_g0.5_fe.npz", allow_pickle=True)
    phi, node = z["pop_fe"], z["pop_fe_node"]
    H = pd.DataFrame({"node": node, "phi": phi})
    H = H[np.isfinite(H.phi)]
    H["modality"] = nodes.modality.to_numpy()[H.node]
    pb = nodes.params_b.to_numpy()[H.node]
    ok = np.isfinite(pb) & (pb > 0) & (pb < 2000)
    H["log_params"] = np.where(ok, np.log(np.where(ok, pb, 1)), 0)
    H["params_na"] = (~ok).astype(float)
    H["major_org"] = nodes.major_org.to_numpy()[H.node].astype(float)
    H = H.merge(q[["node", "bench_z"]], on="node", how="left")
    H["bench_na"] = H.bench_z.isna().astype(float)
    H["bench_z"] = H.bench_z.fillna(0)
    groups = {"quality (bench)": ["bench_z", "bench_na"], "size": ["log_params", "params_na"], "brand": ["major_org"]}
    mod = pd.get_dummies(H.modality.astype(str), prefix="mod", drop_first=True, dtype=float)

    def r2(cols):
        X = pd.concat([mod, H[cols]], axis=1) if cols else mod
        return sm.OLS(H.phi, sm.add_constant(X)).fit().rsquared

    base = r2([])
    allc = sum(groups.values(), [])
    full = r2(allc)
    # Shapley share of R2 across the three groups (modality dummies always in)
    from itertools import combinations
    names = list(groups)
    shap = {g: 0.0 for g in names}
    from math import factorial
    for g in names:
        others = [h for h in names if h != g]
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                w = factorial(k) * factorial(len(names) - k - 1) / factorial(len(names))
                cols = sum((groups[h] for h in S), [])
                shap[g] += w * (r2(cols + groups[g]) - r2(cols))
    fit = sm.OLS(H.phi, sm.add_constant(pd.concat([mod, H[allc]], axis=1))).fit(cov_type="HC1")
    res = pd.DataFrame([dict(component="modality", share_var_phi=base)] +
                       [dict(component=g, share_var_phi=v) for g, v in shap.items()] +
                       [dict(component="unexplained", share_var_phi=1 - full)])
    res.to_csv(paths.TABLES / "p14_phi_decomp.csv", index=False)
    coef = pd.DataFrame({"coef": fit.params, "se": fit.bse}).loc[allc]
    md = ["# P14 hub fixed effects: what part is benchmark quality?", "",
          f"Hubs with FE: {len(H)}; with bench_z: {int((H.bench_na == 0).sum())}. "
          "phi from the popularity mode of the 5-mode mixture (gamma = 0.5, hub FE); OLS with modality dummies.", "",
          "## Share of var(phi) (Shapley over groups, modality first)", "", res.to_markdown(index=False, floatfmt=".3f"), "",
          "## Coefficients (HC1 SE)", "", coef.to_markdown(floatfmt=".3f"), "",
          "Attribution for P15: the benchmark share counts as general quality; brand as popularity; size and the "
          "unexplained remainder are not attributable and bound the fit share (all to popularity = lower bound, "
          "all to fit = upper bound)."]
    (paths.TABLES / "p14_phi_decomp.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.analyze:
        analyze()
