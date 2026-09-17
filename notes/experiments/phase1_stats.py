"""Summarise the Phase 1 emergence statistic from a directory of soup_run batches.

Usage: phase1_stats.py RUNDIR [--png OUT.png]
Prints the number of soups, the fraction transitioned (high-order entropy >= 1 within
the run) with a 95% Wilson interval, quartiles of the transition epoch, and optionally
saves a histogram of transition epochs.
"""

import argparse
import csv
import glob
import json
import pathlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("--png", default=None)
ap.add_argument("--max_epochs", type=int, default=16384)
a = ap.parse_args()

done, epochs = 0, []
for d in sorted(glob.glob(f"{a.rundir}/seed*")):
    d = pathlib.Path(d)
    if not d.is_dir():
        continue
    rows = list(csv.DictReader((d / "log.csv").open()))
    if not rows or int(rows[-1]["epoch"]) < a.max_epochs:
        continue
    soups = len({r["soup"] for r in rows})
    done += soups
    if (d / "transition.json").exists():
        t = json.load((d / "transition.json").open())["epoch"]
        epochs += [e for e in t if e is not None]
n = len(epochs)
p = n / done
z = 1.96
centre = (p + z * z / (2 * done)) / (1 + z * z / done)
half = z * np.sqrt(p * (1 - p) / done + z * z / (4 * done * done)) / (1 + z * z / done)
print(
    f"soups {done}, transitioned {n}, fraction {p:.3f} (95% {centre - half:.3f}-{centre + half:.3f})"
)
if epochs:
    q = np.percentile(epochs, [25, 50, 75])
    print(f"transition epoch quartiles {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}")
if a.png and epochs:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
    ax.hist(epochs, bins=np.arange(0, a.max_epochs + 1, 512), color="#4c78a8")
    ax.set_xlabel("epoch of transition (high-order entropy ≥ 1)")
    ax.set_ylabel("runs")
    ax.set_title(
        f"{n} of {done} soups transitioned within {a.max_epochs} epochs ({100 * p:.0f}%)"
    )
    fig.tight_layout()
    fig.savefig(a.png)
