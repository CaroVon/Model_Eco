"""Single source of truth for paths: code lives in /home, data on the shared space."""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "configs" / "config.yaml").read_text())

DATA = Path(os.environ.get("ACMEC_DATA_ROOT", CFG["data_root"]))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
FIGURES = DATA / "results" / "figures"
TABLES = DATA / "results" / "tables"
LOGS = Path(CFG["log_root"])
OUTPUTS = Path(CFG["output_root"])


def ensure() -> None:
    for p in (RAW, INTERIM, PROCESSED, EXTERNAL, FIGURES, TABLES, LOGS, OUTPUTS):
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for k in ("REPO", "DATA", "RAW", "INTERIM", "PROCESSED", "EXTERNAL", "FIGURES", "TABLES", "LOGS", "OUTPUTS"):
        p = globals()[k]
        print(f"{k:10s} {p}  exists={p.exists()} writable={os.access(p, os.W_OK)}")
