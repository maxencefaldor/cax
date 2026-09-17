"""Bit-exact comparison of cax.cs.bff.run against cubff's Evaluate on random tapes."""

import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import opcode_table_from_string, run

HARNESS = os.environ.get("CUBFF_BIN", "cubff/bin") + "/harness_"


def oracle(tapes: np.ndarray, steps: int, heads: bool):
    out = subprocess.run(
        [HARNESS + ("bff" if heads else "noheads"), str(steps)],
        input=tapes.astype(np.uint8).tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    rec = np.frombuffer(out, dtype=np.uint8).reshape(len(tapes), 132)
    return rec[:, :128].copy(), rec[:, 128:].copy().view(np.uint32).ravel()


def sample(rng, n, kind):
    if kind == "uniform":
        return rng.integers(0, 256, (n, 128), dtype=np.uint8)
    # enriched: 50% instructions, extra zeros, to exercise loops and brackets
    ops = np.frombuffer(b"[]+-.,<>{}", dtype=np.uint8)
    t = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    m = rng.random((n, 128))
    t = np.where(m < 0.5, rng.choice(ops, (n, 128)), t)
    t = np.where((m >= 0.5) & (m < 0.6), 0, t)
    return t.astype(np.uint8)


table = opcode_table_from_string()
rng = np.random.default_rng(0)
total_fail = 0
for heads in (False, True):
    for kind in ("uniform", "enriched"):
        for steps in (8192,):
            n = 4096
            tapes = sample(rng, n, kind)
            ref_t, ref_ops = oracle(tapes, steps, heads)
            t0 = time.time()
            out_t, out_steps, out_ops, _ = run(
                jnp.asarray(tapes), table, num_steps=steps, heads_from_tape=heads
            )
            out_t = np.asarray(jax.block_until_ready(out_t))
            dt = time.time() - t0
            bad = np.any(out_t != ref_t, axis=1)
            bad_ops = np.asarray(out_ops) != ref_ops
            nfail = int(bad.sum() + bad_ops.sum())
            total_fail += nfail
            print(
                f"heads={heads} kind={kind} steps={steps}: tape mismatches={int(bad.sum())}/{n} ops mismatches={int(bad_ops.sum())}  "
                f"mean steps={float(np.mean(np.asarray(out_steps))):.1f} full-budget={float(np.mean(np.asarray(out_steps) == steps)):.3f}  jax {dt:.1f}s"
            )
            if nfail:
                i = int(np.argmax(bad | bad_ops))
                print(" first bad idx", i, "ops", int(out_ops[i]), int(ref_ops[i]))
                print(" in ", tapes[i].tobytes().hex())
                print(" ref", ref_t[i].tobytes().hex())
                print(" out", out_t[i].tobytes().hex())
print("TOTAL FAILURES", total_fail)
