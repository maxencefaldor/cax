"""Mutational robustness of replicators: for each program with detector score >= threshold,
apply every single-byte substitution at the positions it occupies (or a random sample of
them) and measure the fraction of mutants that keep a score >= threshold, and the mean
score change. Programs come from a CSV produced by minimal_replicator.py or from a
checkpoint (--npz / --dat), taking the top-scoring tapes.
Usage: robustness.py OUT.csv --from_csv found.csv [--control flip] [--dialect swap] [--heads]
       [--mutants 256] [--threshold 8]
"""

import argparse, csv
import numpy as np, jax, jax.numpy as jnp
from cax.cs.bff import (
    replication_score,
    sample_partners,
    opcode_table_from_string,
    opcode_table_swap_heads,
    parse,
    unparse,
)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--from_csv")
ap.add_argument("--dat")
ap.add_argument("--npz")
ap.add_argument("--control", default="matched")
ap.add_argument("--dialect", default="two")
ap.add_argument("--heads", action="store_true")
ap.add_argument("--mutants", type=int, default=256)
ap.add_argument("--threshold", type=int, default=8)
ap.add_argument("--top", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
table = opcode_table_from_string() if a.dialect == "two" else opcode_table_swap_heads()
rng = np.random.default_rng(a.seed)

programs = []
if a.from_csv:
    rows = [r for r in csv.reader(open(a.from_csv))]
    rows.sort(key=lambda r: -int(r[3]))
    for r in rows[: a.top]:
        b = np.asarray(parse(r[4], opcode_table=table))
        p = np.zeros(64, np.uint8)
        p[: len(b)] = b
        programs.append(p)
else:
    if a.dat:
        raw = np.fromfile(a.dat, dtype=np.uint8)
        n = int(raw[8:16].view(np.uint64)[0])
        soup = raw[24:].reshape(n, 64)
    else:
        soup = np.load(a.npz)["soup"]
    uniq, counts = np.unique(soup, axis=0, return_counts=True)
    programs = list(uniq[np.argsort(-counts)[: a.top]])
programs = np.stack(programs)
base = np.asarray(
    replication_score(
        jnp.asarray(programs),
        sample_partners(jax.random.key(1), len(programs)),
        table,
        heads_from_tape=a.heads,
        control=a.control,
    )
)
with open(a.out, "a") as f:
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(
            [
                "control",
                "dialect",
                "program",
                "score",
                "mutants",
                "frac_kept",
                "mean_delta",
                "frac_zero",
            ]
        )
    for p, s in zip(programs, base):
        if s < a.threshold:
            continue
        pos = rng.integers(0, 64, a.mutants)
        val = rng.integers(0, 256, a.mutants).astype(np.uint8)
        muts = np.tile(p, (a.mutants, 1))
        muts[np.arange(a.mutants), pos] = val
        ms = np.asarray(
            replication_score(
                jnp.asarray(muts),
                sample_partners(jax.random.key(2), a.mutants),
                table,
                heads_from_tape=a.heads,
                control=a.control,
            )
        )
        w.writerow(
            [
                a.control,
                a.dialect,
                unparse(p, opcode_table=table),
                int(s),
                a.mutants,
                round(float((ms >= a.threshold).mean()), 3),
                round(float((ms - s).mean()), 2),
                round(float((ms == 0).mean()), 3),
            ]
        )
        f.flush()
        print(
            unparse(p, opcode_table=table),
            int(s),
            round(float((ms >= a.threshold).mean()), 3),
            flush=True,
        )
