"""P1: list repo files, download only what is needed, write interim/meta.parquet (plan 4.2).

Usage: python -m hfpa.data.download --list            # just list files
       python -m hfpa.data.download --file <name>     # download one file + print schema
"""
import argparse

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from hfpa import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--file")
    a = ap.parse_args()
    repo = paths.CFG["dataset"]["repo"]
    if a.list or not a.file:
        for f in list_repo_files(repo, repo_type="dataset"):
            print(f)
        return
    paths.ensure()
    p = hf_hub_download(repo, a.file, repo_type="dataset", local_dir=paths.RAW)
    print(p)
    print(pq.read_schema(p))


if __name__ == "__main__":
    main()
