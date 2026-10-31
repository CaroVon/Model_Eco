"""How much of the hub fixed effects is not quality? Identical-weight pairs (P12) as a native check.

The quality upper bound of P15 counts every hub FE phi_j as quality. Two repos with identical weight files have
the same quality, so within a pair the FE difference is non-quality (brand, visibility, tool defaults, name).
With phi_j = q_j + e_j and e independent within a pair:
  Var(phi_a - phi_b) = 2 Var(e)  (pair differences demeaned: the average copy-vs-original gap is reported apart)
  s = Var(e) / Var(phi)          share of hub-FE variance that is not quality, noise-corrected with the FE SEs.
Non-quality parts shared by both copies (family hype) stay in q, so s is a lower bound on the non-quality share
and 1 - s an upper bound on the quality share of the hub FE.
Systematic part: all copies are re-uploads by one tool account (unsloth); the mean copy-minus-original gap is a
publisher premium for identical weights, i.e. non-quality. Its share of var(phi) is gap^2 * Var(1{publisher})
over the hubs (s_sys); the total lower bound is s_sys + s.
Var(phi) and the FE noise use well-identified hubs only (FE SE < 1; a few hubs rarely sampled in the 10% sample
have huge SEs).
FE come from the P15 plan-A conditional logit with hub FE (all feature groups, as in the Shapley upper bound).
Adjusted shares: linear approximation, the FE part of the Shapley quality share is split by s (non-quality part
to popularity); the coarse partition C of p15_shapley_fe is the starting point.

usage: python scripts/p17_mirror_quality.py
Writes results/tables/p17_mirror_quality.md / .csv and p17_mirror_pairs_fe.csv.
"""
import importlib.util

import numpy as np
import pandas as pd

from hfpa import paths

_s = importlib.util.spec_from_file_location("p15", paths.REPO / "scripts" / "p15_decomp.py")
P15 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P15)
N_BOOT = 2000
SE_MAX = 1.0


def share_nonquality(d, var_phi, noise_phi):
    """Noise-corrected Var(e) / Var(q + e) from pair differences d (with their sampling variances)."""
    ve = (np.var(d.dphi, ddof=1) - d.v_noise.mean()) / 2
    return ve / (var_phi - noise_phi)


if __name__ == "__main__":
    rows, ch = P15.load()
    r, g = P15.sample(rows, ch, "planA")
    feats = P15.drop_constant(r, P15.ALL)
    res = P15.fit(r, ch, feats, g, fe=True, cluster=False)
    F = len(feats)
    phi, se = res["coef"][F:], res["se"][F:]
    hub = r.loc[r.fe > 0, ["fe", "cand_id"]].drop_duplicates()
    assert hub.fe.is_unique
    H = pd.DataFrame({"node": hub.cand_id.to_numpy(), "phi": phi[hub.fe.to_numpy() - 1], "se": se[hub.fe.to_numpy() - 1]})
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet", columns=["model_id"])
    H["model_id"] = nodes.model_id.to_numpy()[H.node]
    H.to_parquet(paths.PROCESSED / "choices" / P15.SPEC / "p17_hub_fe.parquet", index=False)
    node_phi = H.set_index("node")
    Hw = H[H.se < SE_MAX]
    var_phi, noise_phi = Hw.phi.var(ddof=1), (Hw.se ** 2).mean()

    # identical-weight pairs: name mirrors with matching hashes + declared-base pairs with identical weights
    mp = pd.read_csv(paths.TABLES / "p12_mirror_pairs.csv")
    hp = pd.read_csv(paths.TABLES / "p12_hash_pairs.csv")
    P = pd.concat([mp.loc[mp.hash_match == "identical", ["orig", "cpy"]].rename(columns={"orig": "a", "cpy": "b"}),
                   hp.loc[hp["match"] == "identical", ["parent_idx", "child_idx"]].rename(columns={"parent_idx": "a", "child_idx": "b"})])
    P = P.drop_duplicates()
    P = P[P.a.isin(node_phi.index) & P.b.isin(node_phi.index)].copy()
    P["a_id"], P["b_id"] = nodes.model_id.to_numpy()[P.a], nodes.model_id.to_numpy()[P.b]
    P["phi_a"], P["phi_b"] = node_phi.phi.loc[P.a].to_numpy(), node_phi.phi.loc[P.b].to_numpy()
    P["dphi"] = P.phi_b - P.phi_a  # copy (declares a as base) minus original
    P["v_noise"] = node_phi.se.loc[P.a].to_numpy() ** 2 + node_phi.se.loc[P.b].to_numpy() ** 2
    P.to_csv(paths.TABLES / "p17_mirror_pairs_fe.csv", index=False)

    s = share_nonquality(P, var_phi, noise_phi)
    rng = np.random.default_rng(0)
    boot = [share_nonquality(P.iloc[rng.integers(0, len(P), len(P))], var_phi, noise_phi) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    s_raw = np.var(P.dphi, ddof=1) / 2 / var_phi
    pub = sorted(set(P.b_id.str.split("/").str[0]))
    ind = Hw.model_id.str.split("/").str[0].isin(pub).to_numpy(float)
    gap = P.dphi.mean()
    s_sys = gap ** 2 * ind.var() / (var_phi - noise_phi)
    boot_sys = []
    for _ in range(N_BOOT):
        Pb = P.iloc[rng.integers(0, len(P), len(P))]
        boot_sys.append(Pb.dphi.mean() ** 2 * ind.var() / (var_phi - noise_phi) + share_nonquality(Pb, var_phi, noise_phi))
    lo_t, hi_t = np.percentile(boot_sys, [2.5, 97.5])

    # adjusted shares (partition C, quality includes hub FE)
    S = pd.read_csv(paths.TABLES / "p15_shapley_fe.csv")
    S = S[S.partition == "C: visibility and availability separate"].set_index("group").share
    S0 = pd.read_csv(paths.TABLES / "p15_shapley.csv")
    S0 = S0[(S0["sample"] == "planA") & (S0.partition == "seven groups")].set_index("group").share
    Q = pd.read_csv(paths.TABLES / "p15_shapley_fe.csv")
    Q = Q[Q.partition == "seven groups"].set_index("group").share
    fe_part = Q["general quality"] - S0["general quality"]  # quality share added by the hub FE

    def adjusted(sv):
        sv = float(np.clip(sv, 0, 1))
        fit = S["fit"] - sv * fe_part
        pop = S["popularity"] + sv * fe_part
        # the publisher premium is a tool account's premium: also reported with the non-quality part as availability
        return dict(fit=fit, popularity=pop, ratio=fit / pop, ratio_if_availability=fit / S["popularity"])
    out = [dict(quantity="pairs with both repos among the hubs", value=len(P)),
           dict(quantity=f"hubs with FE SE < {SE_MAX} (of {len(H)})", value=len(Hw)),
           dict(quantity="var(phi) over these hubs", value=var_phi), dict(quantity="mean FE sampling variance", value=noise_phi),
           dict(quantity=f"publisher of the copies: {', '.join(pub)}; hubs from it", value=int(ind.sum())),
           dict(quantity="mean copy - original phi", value=P.dphi.mean()),
           dict(quantity="corr(phi_a, phi_b)", value=np.corrcoef(P.phi_a, P.phi_b)[0, 1]),
           dict(quantity="s raw (no noise correction)", value=s_raw),
           dict(quantity="s non-quality share of var(phi)", value=s), dict(quantity="s 95% bootstrap lo", value=lo),
           dict(quantity="s 95% bootstrap hi", value=hi), dict(quantity="s_sys publisher premium share", value=s_sys),
           dict(quantity="s_total = s_sys + s", value=s_sys + s), dict(quantity="s_total 95% lo", value=lo_t),
           dict(quantity="s_total 95% hi", value=hi_t), dict(quantity="hub-FE part of quality share", value=fe_part)]
    for tag, sv in [("s_total", s_sys + s), ("s_total lo", lo_t), ("s_total hi", hi_t)]:
        for k, v in adjusted(sv).items():
            out.append(dict(quantity=f"adjusted upper bound ({tag}): {k}", value=v))
    out = pd.DataFrame(out)
    out.to_csv(paths.TABLES / "p17_mirror_quality.csv", index=False)
    md = ["# P17 how much of the hub FE is not quality: identical-weight pairs", "",
          "phi from the P15 plan-A conditional logit with hub FE. s = noise-corrected Var(e)/Var(phi) from demeaned "
          "pair differences; lower bound on the non-quality share (parts shared within a pair count as quality). "
          "s_sys: publisher premium of identical-weight copies (all by unsloth) times the spread of that publisher "
          "over the hubs. Adjusted bound: partition C of p15_shapley_fe, the hub-FE part of the quality share split by "
          "s_total (linear approximation); non-quality part to popularity (ratio) or to availability (ratio_if_availability).", "", out.to_markdown(index=False, floatfmt=".4g"), "", "## Pairs", "",
          P[["a_id", "b_id", "phi_a", "phi_b", "dphi"]].to_markdown(index=False, floatfmt=".3f")]
    (paths.TABLES / "p17_mirror_quality.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
