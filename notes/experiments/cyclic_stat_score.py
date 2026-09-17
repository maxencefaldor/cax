"""Score every finished batch of the cyclic-control statistic with the detector.

Usage: cyclic_stat_score.py RUNDIR [--sample 512]
For each soup with a final checkpoint: the fraction of sampled tapes scoring >= 60 at
epochs 8192 and 16384; a soup counts as transitioned when that fraction is >= 0.5.
"""

import argparse
import glob
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import opcode_table_from_string, replication_score, sample_partners

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("--sample", type=int, default=512)
a = ap.parse_args()
table = opcode_table_from_string()
partners = sample_partners(jax.random.key(1), a.sample)
rng = np.random.default_rng(0)
done, by_epoch = 0, {8192: 0, 16384: 0}
for d in sorted(glob.glob(f"{a.rundir}/seed*")):
    d = pathlib.Path(d)
    if not d.is_dir() or not list(d.glob("ck_*_final.npz")):
        continue
    for epoch in by_epoch:
        soups = np.load(d / f"ck_{epoch:09d}.npz")["soup"]
        for soup in soups:
            idx = rng.choice(soup.shape[0], a.sample, replace=False)
            score = np.asarray(
                replication_score(
                    jnp.asarray(soup[idx]), partners, table, control="cyclic"
                )
            )
            by_epoch[epoch] += int(np.mean(score >= 60) >= 0.5)
    done += len(soups)
for epoch, n in by_epoch.items():
    p = n / done if done else 0.0
    half = 1.96 * np.sqrt(p * (1 - p) / done) if done else 0.0
    print(
        f"epoch {epoch}: {n} of {done} soups replicator-dominated, {p:.3f} ± {half:.3f}"
    )
