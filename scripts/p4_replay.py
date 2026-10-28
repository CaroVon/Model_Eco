"""Stage P4: time replay and choice data (plan 6.5).

usage: python scripts/p4_replay.py --spec main_strat [--frac 0.1]
Writes processed/choices/<spec>[_fracX]/{rows/YYYY-MM.parquet, choices.parquet} and a check report.
"""
import argparse
import json

import numpy as np
import pandas as pd

from hfpa import paths
from hfpa.data.build import T0
from hfpa.replay.replay import Replay

R = paths.CFG["replay"]
SPECS = {
    "main_strat": dict(sampling="stratified", hub_k=R["hub_k"], s1=R["s1"], s0=R["s0"]),
    "main_unif": dict(sampling="uniform", s=R["uniform_s"], hub_k=R["hub_k"]),
}
MAIN = ["finetune", "adapter", "unspecified"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="main_strat", choices=SPECS)
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    name = a.spec + (f"_frac{a.frac:g}" if a.frac < 1 else "")
    out = paths.PROCESSED / "choices" / name
    out.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "birth", "modality", "modality_code", "author_code", "root"])
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet")
    rn = pd.DataFrame({"birth": nodes.birth, "modality": nodes.modality_code, "author": nodes.author_code,
                       "root": nodes.root})
    re_ = pd.DataFrame({"child": E.child_idx, "parent": E.parent_idx, "rel_type": E.rel_type, "weight": E.weight})
    spec = {**SPECS[a.spec], "t_start": int(T0.timestamp()), "sample_frac": a.frac, "seed": paths.CFG["seed"],
            "recent_windows": tuple(R["recent_windows_days"])}
    r = Replay(rn, re_, spec)
    ch, _ = r.run(out_dir=out, progress=True)
    n_hold = int(np.ceil(len(ch) * R["holdout_frac"]))
    ch["holdout"] = np.arange(len(ch)) >= len(ch) - n_hold
    ch["modality_name"] = ch.modality.map(dict(zip(nodes.modality_code, nodes.modality)))
    ch.to_parquet(out / "choices.parquet")

    # --- checks (plan 6.5) ---
    rep = {"spec": name, "n_choices": len(ch), "n_modality_mismatch_dropped": r.n_mod_mismatch,
           "n_hubs": len(r.hubs), "choices_by_modality": ch.modality_name.value_counts().to_dict(),
           "mean_cands": float(ch.n_cands.mean()) if len(ch) else 0}
    direct = E[E.rel_type.isin(MAIN)].parent.value_counts()
    deg_direct = np.zeros(len(nodes), np.int64)
    deg_direct[pd.Index(nodes.model_id).get_indexer(direct.index)] = direct.to_numpy()
    rep["final_deg_equals_P2"] = bool(np.array_equal(r.deg, deg_direct))
    rep["final_deg_equals_internal"] = bool(np.array_equal(r.deg, r.final_deg))
    one = True
    nrows = 0
    for f in sorted((out / "rows").glob("*.parquet")):
        x = pd.read_parquet(f, columns=["choice_id", "chosen"])
        nrows += len(x)
        one &= bool(x.groupby("choice_id").chosen.sum().eq(1).all())
    rep["exactly_one_chosen"] = one
    rep["n_rows"] = nrows
    # spot-check 5 choices: chosen parent's degree at t by hand
    rng = np.random.default_rng(0)
    birth = nodes.birth.to_numpy()
    me = E[E.rel_type.isin(MAIN)]
    spots = []
    for k in rng.choice(len(ch), 5, replace=False):
        c = ch.iloc[k]
        hand = int(((me.parent_idx == c.parent) & (birth[me.child_idx] < c.t)).sum())
        month = pd.Timestamp(int(c.t), unit="s").strftime("%Y-%m")
        x = pd.read_parquet(out / "rows" / f"{month}.parquet", filters=[("choice_id", "==", int(c.choice_id))])
        rec = int(x.loc[x.chosen == 1, "deg"].iloc[0])
        spots.append(dict(choice_id=int(c.choice_id), parent=nodes.model_id.iloc[int(c.parent)],
                          t=str(pd.Timestamp(int(c.t), unit="s")), deg_hand=hand, deg_recorded=rec, ok=hand == rec))
    rep["spot_checks"] = spots
    (out / "checks.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
