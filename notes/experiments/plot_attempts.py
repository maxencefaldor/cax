"""Plot the kernel hill-climb: one point per attempt, wall-clock on the x-axis.

Usage: python plot_attempts.py [OUT.png]   (reads results/kernel_attempts.csv)
"""

import csv
import datetime
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).parent
rows = list(csv.DictReader(open(here / "results" / "kernel_attempts.csv")))
out = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else here / "results" / "kernel_attempts.png"
)
times = [datetime.datetime.fromisoformat(r["utc"]) for r in rows]
regimes = ["random", "enriched", "mid", "final", "mean_ms"]
fig, ax = plt.subplots(figsize=(11, 5.5))
for name in regimes:
    ys = [float(r[name]) for r in rows]
    ax.plot(
        times,
        ys,
        marker="o",
        lw=2 if name == "mean_ms" else 1,
        label=name.replace("_ms", ""),
    )
for t, r in zip(times, rows):
    if r["exact"] != "True":
        ax.annotate("not exact", (t, float(r["mean_ms"])), fontsize=7, color="red")
    if "invalid" in r["description"]:
        ax.annotate(
            "invalid (shared GPU)", (t, float(r["mean_ms"])), fontsize=7, color="gray"
        )
ax.axhline(10.3, ls="--", c="gray", lw=1)
ax.text(times[0], 10.3, " cubff, random", va="bottom", fontsize=8, color="gray")
ax.axhline(35.0, ls="--", c="gray", lw=1)
ax.text(times[0], 35.0, " cubff, transitioned", va="bottom", fontsize=8, color="gray")
ax.set_ylabel("ms per 65,536 pairs × 8192 steps (H100)")
ax.set_xlabel("attempt time (UTC)")
ax.set_title("BFF Pallas kernel hill-climb")
ax.set_ylim(0, None)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out, "with", len(rows), "attempts")
