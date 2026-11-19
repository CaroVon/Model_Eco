import numpy as np
import pandas as pd

from hfpa.models.clogit import CLogit, to_padded
from hfpa.replay.replay import DAY, Replay
from hfpa.sim.synthetic import generate


def toy():
    # 0,1 legacy roots at day 0; 2 (day 1) <- 0; 3 (day 2) <- 0; 4 (day 40) <- 2 (author of 3); 5 (day 41) <- 0 quant
    nodes = pd.DataFrame({"birth": np.array([0, 0, 1, 2, 40, 41]) * DAY, "modality": 0,
                          "author": [0, 1, 2, 3, 3, 4], "root": [0, 1, 0, 0, 0, 0]})
    edges = pd.DataFrame({"child": [2, 3, 4, 5], "parent": [0, 0, 2, 0],
                          "rel_type": ["finetune", "adapter", "finetune", "quantized"], "weight": 1.0})
    return nodes, edges


def test_toy_state():
    nodes, edges = toy()
    r = Replay(nodes, edges, {"hub_k": 1, "s1": 5, "s0": 5})
    ch, rows = r.run()
    assert list(ch.child) == [2, 3, 4]
    assert np.array_equal(r.deg, r.final_deg) and r.deg[0] == 2 and r.quant_deg[0] == 1
    assert rows.groupby("choice_id").chosen.sum().eq(1).all()
    last = rows[rows.choice_id == 2].set_index("cand_id")
    assert last.loc[0, "deg"] == 2 and last.loc[0, "recent30"] == 0 and last.loc[0, "recent90"] == 2
    assert last.loc[2, "age_days"] == 39 and last.loc[2, "chosen"] == 1
    assert last.loc[3, "same_author"] == 1 and last.loc[0, "author_reuse"] == 1  # author 3 used parent 0 before
    assert last.loc[2, "author_family"] == 1 and last.loc[0, "fam_size"] == 3  # nodes 0,2,3 born before day 40
    assert last.loc[0, "fam_recent"] == 0


def test_clogit_recovers_pa():
    nodes, edges, truth = generate(n=8000, alpha=1.0, beta0=-1.0, seed=1)
    ch, rows = Replay(nodes, edges, {"hub_k": 50, "s1": 20, "s0": 20}).run()
    D = to_padded(rows, ["has_deg", "log_deg"])
    res = CLogit(2, device="cpu").fit(D)
    a, se = res["coef"][1], res["se"][1]
    assert abs(a - 1.0) < 3 * se + 0.05, (a, se)


def test_pps_inclusion_matches_pi():
    """Empirical inclusion frequency of each node ~ pi = exp(log_q); expected set size ~ pps_size."""
    nodes, edges, _ = generate(n=1500, alpha=1.0, beta0=-1.0, seed=3)
    # repeated draws of the last choice (same state, different seeds)
    R = 200
    hits, pis, sizes = {}, {}, []
    for seed in range(R):
        ch, rows = Replay(nodes, edges, {"sampling": "pps", "hub_k": 0, "pps_size": 60, "seed": seed}).run()
        last = rows[rows.choice_id == ch.choice_id.max()]
        sizes.append((last.chosen == 0).sum())
        for j, lq, c in zip(last.cand_id, last.log_q, last.chosen):
            if c:
                continue
            hits[j] = hits.get(j, 0) + 1
            pis[j] = np.exp(lq)
    sizes = np.array(sizes)
    assert abs(sizes.mean() - 60) < 3, sizes.mean()
    # pi = 1 nodes always present; high-pi nodes appear about pi of the time
    certain = [j for j, p in pis.items() if p >= 1]
    assert all(hits[j] >= R - 1 for j in certain)  # chosen parent may replace one draw
    mid = [j for j, p in pis.items() if 0.2 < p < 0.9]
    if mid:
        err = np.mean([hits[j] / R - pis[j] for j in mid])
        assert abs(err) < 0.05, err


def test_clogit_recovers_pa_all_q():
    nodes, edges, truth = generate(n=8000, alpha=1.0, beta0=-1.0, seed=1)
    for spec in ({"sampling": "pps", "hub_k": 0, "pps_size": 60},
                 {"sampling": "uniform", "s": 60, "hub_k": 0},
                 {"hub_k": 50, "s1": 20, "s0": 20}):
        ch, rows = Replay(nodes, edges, spec).run()
        D = to_padded(rows, ["has_deg", "log_deg"])
        res = CLogit(2, device="cpu").fit(D)
        a, se = res["coef"][1], res["se"][1]
        assert abs(a - 1.0) < 3 * se + 0.05, (spec, a, se)


def test_history_included_with_certainty():
    nodes, edges = toy()
    for sampling in ("pps", "uniform", "stratified"):
        r = Replay(nodes, edges, {"sampling": sampling, "hub_k": 0, "s": 1, "s1": 1, "s0": 1, "pps_size": 1,
                                  "include_hist": True})
        ch, rows = r.run()
        last = rows[rows.choice_id == 2].set_index("cand_id")  # author 3 used parent 0 before
        assert 0 in last.index and last.loc[0, "log_q"] == 0 and last.loc[0, "stratum"] == 4


def test_fe_time_ex_ante():
    nodes, edges = toy()
    r = Replay(nodes, edges, {"fe_time": 3 * DAY, "fe_k": 5})
    assert r.fe_idx[0] == 1 and r.fe_idx[2] == 0  # node 2's child is born on day 40
