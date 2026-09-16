"""Run a BFF soup in CAX and log the paper's complexity signal.

Usage: python soup_run.py OUTDIR --seed S --num_programs N [--max_epochs E] [--log_every 64]
       [--stop_at 3.0] [--mutation_rate 1/4096] [--heads] [--checkpoint_every 4096]
       [--soups R]
Logs epoch, brotli size, byte entropy, high-order entropy, mean steps per pair, wall time.
With `--soups R`, R independent soups (seeds S, S+1, ...) step together in one kernel
launch per epoch, which is 2-3x cheaper per soup on a GPU; the log gets one row per soup
per log point (column `soup`), checkpoints hold all soups, and a run stops when every
soup has transitioned and run `stop_after` more epochs, or at `max_epochs`.
"""

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cax.cs.bff import (
    BFF,
    byte_entropy,
    compressed_bits_per_byte,
    opcode_table_swap_heads,
)

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--num_programs", type=int, default=8192)
ap.add_argument("--max_epochs", type=int, default=10**9)
ap.add_argument("--log_every", type=int, default=64)
ap.add_argument("--stop_at", type=float, default=3.0)
ap.add_argument(
    "--stop_after", type=int, default=2048, help="epochs to keep running past stop_at"
)
ap.add_argument("--mutation_rate", type=float, default=1 / 4096)
ap.add_argument("--heads", action="store_true", help="heads_from_tape (cubff bff)")
ap.add_argument("--control", default="matched", choices=["matched", "cyclic", "flip"])
ap.add_argument("--dialect", default="two", choices=["two", "swap"])
ap.add_argument("--tape_length", type=int, default=64)
ap.add_argument("--checkpoint_every", type=int, default=4096)
ap.add_argument("--resume", default=None)
ap.add_argument("--soups", type=int, default=1, help="independent soups per process")
a = ap.parse_args()
out = pathlib.Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)

table = opcode_table_swap_heads() if a.dialect == "swap" else None
cs = BFF(
    opcode_table=table,
    mutation_rate=a.mutation_rate,
    heads_from_tape=a.heads,
    control=a.control,
    rngs=nnx.Rngs(a.seed),
)
if a.resume:
    ck = np.load(a.resume)
    soup = jnp.asarray(ck["soup"])
    epoch = int(ck["epoch"])
else:
    soup = cs.init_state(num_programs=a.num_programs, num_soups=a.soups)
    epoch = 0
if soup.ndim == 2:
    soup = soup[None]
R = soup.shape[0]


@nnx.jit(static_argnames=("n",))
def chunk(cs, soup, n):
    def body(cs, soup, _):
        keys = jax.random.split(cs.rngs.pairing(), soup.shape[0])
        perm = jax.vmap(lambda k: jax.random.permutation(k, soup.shape[1]))(keys)
        soup, steps = cs.pair_and_run(soup, perm)
        return soup, steps.mean(axis=-1)

    soup, steps = nnx.scan(
        body,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        length=n,
    )(cs, soup, None)
    return soup, steps.mean(axis=0)  # (soups,)


log = open(out / "log.csv", "a")
if log.tell() == 0:
    log.write("epoch,soup,brotli_bytes,h0,hoe,mean_steps,wall_s\n")
t0 = time.time()
transitioned_at = [None] * R
while epoch < a.max_epochs:
    soup, mean_steps = chunk(cs, soup, a.log_every)
    epoch += a.log_every
    all_soups = np.asarray(soup)
    mean_steps = np.asarray(mean_steps)
    for i in range(R):
        s = all_soups[i]
        h0 = byte_entropy(s)
        bpb = compressed_bits_per_byte(s)
        hoe = h0 - bpb
        log.write(
            f"{epoch},{i},{int(bpb * s.size / 8)},{h0:.4f},{hoe:.4f},{float(mean_steps[i]):.1f},{time.time() - t0:.1f}\n"
        )
        if hoe >= a.stop_at and transitioned_at[i] is None:
            transitioned_at[i] = epoch
            np.savez_compressed(
                out / f"ck_{epoch:09d}_transition_soup{i}.npz", soup=s, epoch=epoch
            )
            (out / "transition.json").write_text(
                json.dumps(
                    {
                        "epoch": transitioned_at,
                        "wall_s": time.time() - t0,
                        "seeds": [a.seed + j for j in range(R)],
                    }
                )
            )
    log.flush()
    if epoch % a.checkpoint_every == 0:
        np.savez_compressed(out / f"ck_{epoch:09d}.npz", soup=all_soups, epoch=epoch)
    if all(t is not None and epoch >= t + a.stop_after for t in transitioned_at):
        break
np.savez_compressed(
    out / f"ck_{epoch:09d}_final.npz", soup=np.asarray(soup), epoch=epoch
)
print("done", epoch, "transition at", transitioned_at)
