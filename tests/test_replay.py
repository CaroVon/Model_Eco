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
