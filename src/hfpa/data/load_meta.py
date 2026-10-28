"""P1: parse the raw CSV (minus modelCard) into interim/meta.parquet (plan 4.2).

List-valued columns are stored in the CSV as Python reprs ("['a', 'b']") and are parsed
with ast.literal_eval. `author` is not a column of the dataset; it is the model_id prefix.
"""
import ast

import pandas as pd

from hfpa import paths

CSV = paths.RAW / "ai_ecosystem_withmodelcards.csv"
LIST_COLS = ["tags", "regions", "licenses", "arxiv_papers", "datasets", "parent_model", "finetune_parent",
             "quantized_parent", "adapter_parent", "merge_parent", "languages"]
SCALAR_COLS = ["model_id", "likes", "trendingScore", "private", "downloads", "pipeline_tag", "library_name",
               "createdAt", "ratelimit_retries", "exception_raised"]


def _parse_list(x):
    if not isinstance(x, str) or not x:
        return []
    try:
        v = ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [x]
    return [str(i) for i in v] if isinstance(v, (list, tuple, set)) else [str(v)]


def load(chunksize=200_000):
    parts = []
    for ch in pd.read_csv(CSV, usecols=SCALAR_COLS + LIST_COLS, chunksize=chunksize, dtype=str,
                          keep_default_na=False, na_values=[""]):
        for c in LIST_COLS:
            ch[c] = ch[c].map(_parse_list)
        parts.append(ch)
        print(f"parsed {sum(len(p) for p in parts):,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    for c in ["likes", "downloads", "ratelimit_retries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trendingScore"] = pd.to_numeric(df["trendingScore"], errors="coerce")
    df["createdAt_raw"] = df["createdAt"]
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce", format="ISO8601")
    df["author"] = df["model_id"].str.split("/", n=1).str[0]
    return df


if __name__ == "__main__":
    paths.ensure()
    df = load()
    out = paths.INTERIM / "meta.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df):,} cols={list(df.columns)}")
