"""Score the tapes of an anchored grid checkpoint with the replication detector.

Usage: weave_score.py CKPT.npz [--anchor 64] [--control cyclic] [--sample 2048]
Every `anchor`-cell stretch along every axis that starts on the anchor lattice is a
tape; a random sample of them is scored as the soup's programs are.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import opcode_table_from_string, replication_score, sample_partners

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--anchor", type=int, default=64)
ap.add_argument("--control", default="cyclic")
ap.add_argument("--sample", type=int, default=2048)
a = ap.parse_args()

memory = np.load(a.ckpt)["memory"]
tapes = []
for axis in range(memory.ndim):
    m = np.moveaxis(memory, axis, -1)
    flat = m.reshape(-1, m.shape[-1])
    for start in range(0, m.shape[-1], a.anchor):
        tapes.append(flat[:, start : start + a.anchor])
tapes = np.concatenate(tapes)
rng = np.random.default_rng(0)
idx = rng.choice(len(tapes), min(a.sample, len(tapes)), replace=False)
table = opcode_table_from_string()
score = np.asarray(
    replication_score(
        jnp.asarray(tapes[idx]),
        sample_partners(jax.random.key(1), len(idx)),
        table,
        control=a.control,
    )
)
print(
    f"{a.ckpt}: {len(tapes)} tapes, sampled {len(idx)}: "
    f"score>=60 {np.mean(score >= 60):.4f}, >=32 {np.mean(score >= 32):.4f}, "
    f">=16 {np.mean(score >= 16):.4f}, max {score.max()}"
)
