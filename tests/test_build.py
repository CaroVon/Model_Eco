import numpy as np
import pandas as pd

from hfpa.data import build
from hfpa.data.build import build_nodes, clean_edges, raw_edges


def meta(n_ghost_kids=3):
    ids = ["org/base", "a/ft1", "b/ft2"] + [f"k{i}/kid" for i in range(n_ghost_kids)]
    t = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"] + ["2023-04-01"] * n_ghost_kids, utc=True)
    tags = [[], ["base_model:finetune:org/base"], ["base_model:org/base"]] + [["base_model:finetune:gone/ghost"]] * n_ghost_kids
    m = pd.DataFrame({"model_id": ids, "createdAt": t.as_unit("us"), "tags": tags,
                      "pipeline_tag": ["text-generation"] * 3 + ["text-to-image"] * n_ghost_kids,
                      "library_name": None, "licenses": [[]] * len(ids), "arxiv_papers": [[]] * len(ids),
                      "downloads": 0, "likes": 0, "trendingScore": 0})
    m["author"] = m.model_id.str.split("/").str[0]
    return m


def test_birth_is_unix_seconds_for_microsecond_timestamps():
    # regression: datetime64[us] read from parquet once gave birth in microseconds -> zero choices in P4
    m = meta()
    E, _, info = clean_edges(m, raw_edges(m))
    nodes, _ = build_nodes(pd.concat([m, info["placeholders"]], ignore_index=True), E)
    assert nodes.birth.iloc[0] == int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
    assert np.all(np.diff(nodes.birth) >= 0)


def test_placeholder_for_large_dangling_parent(monkeypatch):
    monkeypatch.setattr(build, "PLACEHOLDER_MIN", 3)
    m = meta(3)
    E, funnel, info = clean_edges(m, raw_edges(m))
    ph = info["placeholders"]
    assert list(ph.model_id) == ["gone/ghost"] and ph.createdAt.iloc[0] < pd.Timestamp("2023-04-01", tz="UTC")
    assert (E.parent == "gone/ghost").sum() == 3
    nodes, E = build_nodes(pd.concat([m.assign(placeholder=False), ph], ignore_index=True), E)
    g = nodes.set_index("model_id").loc["gone/ghost"]
    assert g.placeholder and g.modality == "image_gen"
    monkeypatch.setattr(build, "PLACEHOLDER_MIN", 4)  # below threshold: dropped as dangling
    E2, _, info2 = clean_edges(m, raw_edges(m))
    assert len(info2["placeholders"]) == 0 and (E2.parent == "gone/ghost").sum() == 0
