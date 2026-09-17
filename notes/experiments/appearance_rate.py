"""Programs tested per replicator found (Knierim et al. 2026, Sec. 3.1) for BFF variants.

For each (control, heads, distribution): sample programs, score them with the detector,
and record the score histogram, the fraction of programs that write at least one byte
when run against a random partner, mean steps, and wall time.
Usage: appearance_rate.py OUT.csv --num 20000 --batch 2000 [--controls matched,cyclic,flip]
       [--heads two,swap] [--dists uniform,cust] [--seed 0]
"""
import argparse, csv, time
import numpy as np, jax, jax.numpy as jnp
from cax.cs.bff import (replication_score, sample_partners, run, opcode_table_from_string,
                        opcode_table_swap_heads, is_instruction)

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--num", type=int, default=20000); ap.add_argument("--batch", type=int, default=2000)
ap.add_argument("--controls", default="matched,cyclic,flip"); ap.add_argument("--heads", default="two,swap")
ap.add_argument("--dists", default="uniform,cust"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

def sample(rng, n, table, dist):
    if dist == "uniform":
        return rng.integers(0, 256, (n, 64), dtype=np.uint8)
    instr = np.array([b for b in range(256) if bool(is_instruction(table[b]))], dtype=np.uint8)
    other = np.array([b for b in range(256) if not bool(is_instruction(table[b]))], dtype=np.uint8)
    coin = rng.random((n, 64)) < 0.5
    return np.where(coin, rng.choice(instr, (n, 64)), rng.choice(other, (n, 64))).astype(np.uint8)

with open(a.out, "a") as f:
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["control", "heads", "dist", "num", "writes_frac", "mean_steps", "ge8", "ge16", "ge32", "ge48", "ge60", "max", "secs", "hist"])
    for control in a.controls.split(","):
        for heads in a.heads.split(","):
            table = opcode_table_from_string() if heads == "two" else opcode_table_swap_heads()
            for dist in a.dists.split(","):
                rng = np.random.default_rng(a.seed)
                scores, writes, steps_all, t0 = [], [], [], time.time()
                for i in range(0, a.num, a.batch):
                    progs = sample(rng, min(a.batch, a.num - i), table, dist)
                    partners = sample_partners(jax.random.key(a.seed * 1000 + i), len(progs))
                    scores.append(np.asarray(replication_score(jnp.asarray(progs), partners, table, control=control)))
                    # does the program write anything when run against a random partner?
                    pair = np.concatenate([progs, np.asarray(partners[:, 0])], axis=1)
                    out, st, _ = run(jnp.asarray(pair), table, control=control)
                    writes.append(np.any(np.asarray(out) != pair, axis=1)); steps_all.append(np.asarray(st))
                s = np.concatenate(scores); wr = np.concatenate(writes); st = np.concatenate(steps_all)
                hist = np.bincount(s, minlength=65)
                row = [control, heads, dist, len(s), round(float(wr.mean()), 4), round(float(st.mean()), 1),
                       int((s >= 8).sum()), int((s >= 16).sum()), int((s >= 32).sum()), int((s >= 48).sum()), int((s >= 60).sum()), int(s.max()),
                       round(time.time() - t0, 1), " ".join(f"{k}:{v}" for k, v in enumerate(hist) if v and k > 0)]
                w.writerow(row); f.flush(); print(row[:13], flush=True)
