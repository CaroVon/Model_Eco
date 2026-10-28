"""Stage P1a: raw CSV -> interim/meta.parquet."""
from hfpa.data.load_meta import CSV, load
from hfpa import paths

if __name__ == "__main__":
    paths.ensure()
    df = load()
    out = paths.INTERIM / "meta.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df):,} cols={list(df.columns)}")
