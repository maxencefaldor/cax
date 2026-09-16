"""The paper's "seeded" experiment: one hand-written replicator in program 0 of a random
soup, 128 epochs; record the high-order entropy at the end (transition if >= 1, or 3)."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cax.cs.bff import BFF, high_order_entropy, opcode_table_swap_heads, parse

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--num_programs", type=int, default=8192)
ap.add_argument("--seeds", type=int, default=50)
ap.add_argument("--first_seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=128)
ap.add_argument("--program", default="[[{.>]-]A]-]>.{[[")
ap.add_argument("--control", default="matched", choices=["matched", "cyclic", "flip"])
ap.add_argument("--dialect", default="two", choices=["two", "swap"])
ap.add_argument("--heads", action="store_true")
a = ap.parse_args()


@nnx.jit(static_argnames=("n",))
def chunk(cs, soup, n):
    def body(cs, soup, _):
        perm = jax.random.permutation(cs.rngs.pairing(), soup.shape[0])
        soup, _ = cs.pair_and_run(soup, perm)
        return soup, None

    soup, _ = nnx.scan(
        body,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        length=n,
    )(cs, soup, None)
    return soup


table = opcode_table_swap_heads() if a.dialect == "swap" else None
prog = np.asarray(parse(a.program, opcode_table=table))
with open(a.out, "a") as f:
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        t0 = time.time()
        cs = BFF(opcode_table=table, control=a.control, heads_from_tape=a.heads, rngs=nnx.Rngs(seed))
        soup = cs.init_state(num_programs=a.num_programs)
        soup = soup.at[0, : len(prog)].set(jnp.asarray(prog))
        soup = chunk(cs, soup, a.epochs)
        hoe = high_order_entropy(np.asarray(soup))
        f.write(f"{seed},{hoe:.4f},{time.time() - t0:.1f}\n")
        f.flush()
