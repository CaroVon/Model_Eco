"""Stage P3: descriptive evidence (plan 6.4)."""
import pandas as pd

from hfpa import paths
from hfpa.data.build import T0
from hfpa.viz import descriptive as d

if __name__ == "__main__":
    E = pd.read_parquet(paths.PROCESSED / "edges" / "edges.parquet")
    nodes = pd.read_parquet(paths.PROCESSED / "nodes" / "nodes.parquet",
                            columns=["model_id", "birth", "birth_ts", "legacy", "root"])
    fits = d.degree_ccdf(E)
    print(fits.to_string())
    conc = d.concentration(E, nodes, T0)
    print(conc.iloc[[0, len(conc) // 2, -1]].to_string())
    d.family_growth(nodes)
    d.head_start(E, nodes, T0)
    s = d.monthly_choices(E, nodes, T0)
    print(s.describe())
