"""Search for the shortest self-replicators of a BFF variant.

Enumerates every program of `--length` bytes over the variant's instruction alphabet
(plus the zero byte, optionally), places it at the start of a zeroed 64-byte tape, and
scores it with the replication detector against random partners. Lengths up to 5 or 6
are exhaustive (10^5 to 10^6 programs); beyond that use --sample.
Usage: minimal_replicator.py OUT.csv --length 5 [--control flip] [--dialect swap]
       [--heads] [--with_zero] [--sample N] [--threshold 4] [--batch 4096]
"""
import argparse, csv, itertools, time
import numpy as np, jax, jax.numpy as jnp
from cax.cs.bff import (replication_score, sample_partners, opcode_table_from_string,
                        opcode_table_swap_heads, is_instruction, unparse)

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--length", type=int, default=5)
ap.add_argument("--control", default="matched"); ap.add_argument("--dialect", default="two")
ap.add_argument("--heads", action="store_true"); ap.add_argument("--with_zero", action="store_true")
ap.add_argument("--sample", type=int, default=0, help="random sample size instead of exhaustive")
ap.add_argument("--threshold", type=int, default=4, help="score at or above which a program is reported")
ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
table = opcode_table_from_string() if a.dialect == "two" else opcode_table_swap_heads()
alphabet = [b for b in range(256) if bool(is_instruction(table[b]))] + ([0] if a.with_zero else [])
rng = np.random.default_rng(a.seed)
if a.sample:
    cands = rng.choice(np.array(alphabet, dtype=np.uint8), (a.sample, a.length))
else:
    cands = np.array(list(itertools.product(alphabet, repeat=a.length)), dtype=np.uint8)
print(f"{len(cands)} candidates of length {a.length} over {len(alphabet)} symbols", flush=True)
t0 = time.time(); found = 0
with open(a.out, "a") as f:
    w = csv.writer(f)
    for i in range(0, len(cands), a.batch):
        c = cands[i:i + a.batch]
        progs = np.zeros((len(c), 64), dtype=np.uint8); progs[:, :a.length] = c
        partners = sample_partners(jax.random.key(a.seed * 7919 + i), len(progs))
        scores = np.asarray(replication_score(jnp.asarray(progs), partners, table, heads_from_tape=a.heads, control=a.control))
        for j in np.nonzero(scores >= a.threshold)[0]:
            w.writerow([a.control, a.dialect, a.length, int(scores[j]), unparse(progs[j][:a.length], opcode_table=table)]); found += 1
        f.flush()
        if (i // a.batch) % 10 == 0:
            print(f"{i + len(c)}/{len(cands)} scored, {found} at score >= {a.threshold}, {time.time() - t0:.0f}s", flush=True)
print("done", found, "found", flush=True)
