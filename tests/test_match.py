import numpy as np
import pandas as pd

from hfpa.features.match import build, node_attrs


def toy():
    # parents 0 (en, text-generation, 7B), 1 (fr, text-to-image, 1B), 2 (en, text-generation, 7B, non-commercial)
    # author 0: child 3 chose 0 at t=1; child 4 (t=5, declares d1, own language "de") is the choice evaluated
    # author 1: children 5, 6, 7 (t=2,3,4) and 8 (t=10, after the choice) trained on d1, all chose parent 1
    nodes = pd.DataFrame({
        "pipeline_tag": ["text-generation", "text-to-image", "text-generation", None, "text-generation",
                         "text-to-image", "text-to-image", "text-to-image", "text-to-image"],
        "params_b": [7, 1, 7, np.nan, np.nan, 1, 1, 1, 1],
        "license_cat": ["apache", "mit", "noncommercial"] + ["none"] * 6,
        "major_org": [False, False, False, True, True, False, False, False, False]})
    langs = [["en"], ["fr"], ["en"], None, ["de"], ["fr"], ["fr"], ["fr"], ["fr"]]
    ds = [None, None, None, None, ["d1"], ["d1"], ["d1"], ["d1"], ["d1"]]
    author = np.array([9, 9, 9, 0, 0, 1, 1, 1, 1])
    full = pd.DataFrame({"child": [3, 5, 6, 7, 4, 8], "parent": [0, 1, 1, 1, 2, 1], "t": [1, 2, 3, 4, 5, 10]})
    ch = pd.DataFrame({"choice_id": [0], "child": [4], "t": [5]}).set_index("choice_id", drop=False)
    rows = pd.DataFrame({"choice_id": [0, 0, 0], "cand_id": [0, 1, 2]})
    return build(rows, ch, node_attrs(nodes, langs, ds), full, author).set_index(rows.cand_id)


def test_history_leave_one_out():
    f = toy()
    assert list(f.hist_has) == [0, 1, 1]  # parent 0 is the only earlier parent: left out for candidate 0
    assert list(f.lang_hist) == [0, 0, 1] and list(f.task_hist) == [0, 0, 1]
    assert f.size_na[0] == 1 and np.isclose(f.size_dist[1], np.log(7)) and f.size_dist[2] == 0


def test_dataset_demand_excludes_child_and_future():
    f = toy()
    assert list(f.ds_has) == [1, 1, 1]
    assert list(f.ds_lang) == [0, 1, 0]  # d1 profile is French; the child's own "de" label is not used
    assert list(f.ds_task) == [0, 1, 0]
    assert np.isclose(f.ds_couse[1], np.log1p(3))  # child 8 comes after the choice
    assert list(f.org_noncomm) == [0, 0, 1]
