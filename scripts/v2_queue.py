"""Resumable runner for the v2 GPU steps (one Slurm job runs steps until its time budget is used).

Steps are run in order; a step is skipped if done (<state>/<name>.done) or locked by another running job
(<state>/<name>.lock with a live Slurm job id). A failed step writes <name>.failed (and is not retried
automatically). Several runners (protected QOS, general queue) can work through the same list in parallel.
usage: python scripts/v2_queue.py --budget 100      (minutes; no new step starts after the budget)
       python scripts/v2_queue.py --status
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from hfpa import paths

STATE = paths.PROCESSED / "choices" / "v2_queue"
PY = [sys.executable]
M = "scripts/v2_main.py"
STEPS = [
    ("verify_fe", ["scripts/v2_verify_fe.py"]),
    ("clogit_pps", [M, "--spec", "v2_pps", "--part", "clogit"]),
    ("shap_nofe_pps", [M, "--spec", "v2_pps", "--part", "shapley", "--variant", "nofe"]),
    ("clogit_strat", [M, "--spec", "v2_strat", "--part", "clogit"]),
    ("shap_nofe_strat", [M, "--spec", "v2_strat", "--part", "shapley", "--variant", "nofe"]),
    ("clogit_unif", [M, "--spec", "v2_unif", "--part", "clogit"]),
    ("shap_nofe_unif", [M, "--spec", "v2_unif", "--part", "shapley", "--variant", "nofe"]),
    ("shap_fe_pps", [M, "--spec", "v2_pps", "--part", "shapley", "--variant", "fe"]),
    ("boot_pps_0", [M, "--spec", "v2_pps", "--part", "boot", "--task", "0", "--reps", "2", "--fe"]),
    ("networks_sir", ["scripts/v2_networks.py", "--part", "sir"]),
    ("nested", ["scripts/v2_retention.py", "--part", "nested"]),
    ("classify", ["scripts/v2_welfare.py", "--classify"]),
    ("mixture_planA_fit", ["scripts/v2_mixture.py", "--sample", "planA", "--fit"]),
]
STEPS += [(f"boot_pps_{i}", [M, "--spec", "v2_pps", "--part", "boot", "--task", str(i), "--reps", "2", "--fe"])
          for i in range(1, 25)]
STEPS += [(f"boot_{s}_{i}", [M, "--spec", f"v2_{s}", "--part", "boot", "--task", str(i), "--reps", "2"])
          for s in ("strat", "unif") for i in range(10)]
STEPS += [
    ("mixture_planA_sim", ["scripts/v2_mixture.py", "--sample", "planA", "--sim"]),
    ("mixture_decisions_fit", ["scripts/v2_mixture.py", "--sample", "decisions", "--fit"]),
    ("placebo", ["scripts/v2_bounds.py", "--placebo"]),
    ("robust_signal", ["scripts/v2_robust.py", "--signal"]),
    ("robust_text", ["scripts/v2_robust.py", "--text"]),
]
# steps that need outputs of earlier steps
NEEDS = {"shap_fe_pps": [], "networks_sir": ["clogit_pps", "q_ext"], "nested": ["clogit_pps"],
         "classify": ["clogit_pps"], "placebo": ["placebo_feats"]}
for n, _ in STEPS:
    if n.startswith("boot_pps"):
        NEEDS[n] = ["clogit_pps", "shap_nofe_pps"]
    elif n.startswith("boot_strat"):
        NEEDS[n] = ["clogit_strat", "shap_nofe_strat"]
    elif n.startswith("boot_unif"):
        NEEDS[n] = ["clogit_unif", "shap_nofe_unif"]


def alive(jobid):
    if not jobid:
        return False
    if jobid == os.environ.get("SLURM_JOB_ID"):
        return True
    r = subprocess.run(["squeue", "-h", "-j", jobid], capture_output=True, text=True)
    return bool(r.stdout.strip())


def status():
    for n, _ in STEPS:
        st = "done" if (STATE / f"{n}.done").exists() else "FAILED" if (STATE / f"{n}.failed").exists() else \
            "running" if (STATE / f"{n}.lock").exists() and alive((STATE / f"{n}.lock").read_text().split()[0]) else "todo"
        print(f"{st:8s} {n}")


def run(budget):
    STATE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    me = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    for n, cmd in STEPS:
        if (time.time() - t0) / 60 > budget:
            print("budget used; stopping", flush=True)
            return
        if (STATE / f"{n}.done").exists() or (STATE / f"{n}.failed").exists():
            continue
        if any(not (STATE / f"{d}.done").exists() for d in NEEDS.get(n, [])):
            continue
        lock = STATE / f"{n}.lock"
        if lock.exists() and alive(lock.read_text().split()[0]):
            continue
        lock.write_text(f"{me} {time.time()}")
        print(f"=== {n}: {' '.join(cmd)}", flush=True)
        s = time.time()
        r = subprocess.run(PY + cmd, cwd=paths.REPO)
        lock.unlink(missing_ok=True)
        tag = "done" if r.returncode == 0 else "failed"
        (STATE / f"{n}.{tag}").write_text(f"{me} {time.time() - s:.0f}s")
        print(f"=== {n} {tag} in {(time.time() - s) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=100)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    status() if a.status else run(a.budget)
