"""Benchmark the interpreter on two workloads: a random soup and a transitioned one.

Usage: python bench_kernel.py [--soup CK.npz] [--blocks 32,64] [--epochs 32]
Reports, per workload: `run_kernel` alone on the paired tapes (8192 steps), and the
full epoch through `BFF.pair_and_run` (gather, mutation, kernel, write-back) with
`jax.random.permutation` included; both in ms, medians over `--epochs` calls.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cax.cs.bff import BFF, opcode_table_from_string
from cax.cs.bff.kernel import run_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--soup", default=None, help="npz checkpoint with a `soup` array")
ap.add_argument("--num_programs", type=int, default=131072)
ap.add_argument("--blocks", default="32")
ap.add_argument("--epochs", type=int, default=32)
a = ap.parse_args()

table = opcode_table_from_string()
workloads = {
    "random": jax.random.randint(
        jax.random.key(0), (a.num_programs, 64), 0, 256, dtype=jnp.uint8
    )
}
if a.soup:
    workloads["transitioned"] = jnp.asarray(np.load(a.soup)["soup"])


def median_ms(f, *args):
    f(*args)[0].block_until_ready()
    times = []
    for _ in range(a.epochs):
        t = time.perf_counter()
        f(*args)[0].block_until_ready()
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times))


for name, soup in workloads.items():
    perm = jax.random.permutation(jax.random.key(1), soup.shape[0])
    pairs = soup[perm].reshape(-1, 128)
    for block in (int(b) for b in a.blocks.split(",")):
        f = jax.jit(
            lambda t, block=block: run_kernel(t, table, num_steps=8192, block=block)
        )
        ms = median_ms(f, pairs)
        steps = float(f(pairs)[1].mean())
        print(
            f"{name:12s} kernel block={block:3d}: {ms:7.2f} ms  (mean steps {steps:.0f})",
            flush=True,
        )
    cs = BFF(rngs=nnx.Rngs(0))

    @nnx.jit
    def epoch(cs, soup):
        perm = jax.random.permutation(cs.rngs.pairing(), soup.shape[0])
        return cs.pair_and_run(soup, perm)

    ms = median_ms(lambda s: epoch(cs, s), soup)
    print(f"{name:12s} epoch (pair_and_run):   {ms:7.2f} ms", flush=True)
