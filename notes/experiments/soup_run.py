"""Run a BFF soup in CAX and log the paper's complexity signal.

Usage: python soup_run.py OUTDIR --seed S --num_programs N [--max_epochs E] [--log_every 64]
       [--stop_at 3.0] [--mutation_rate 1/4096] [--heads] [--checkpoint_every 4096]
       [--soups R] [--grid H W --frame_every 16]
Logs epoch, brotli size, byte entropy, high-order entropy, mean steps per pair, the
number of species (distinct instruction skeletons) and the share of the largest, and
wall time. With `--grid H W`, a frame of the grid `render` is saved every `frame_every`
epochs under `frames_soup{i}/`; with `--pairing local` programs pair with their torus
neighbours instead of at random.
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
    BFFEconomy,
    EconomyState,
    byte_entropy,
    compressed_bits_per_byte,
    opcode_table_swap_heads,
    parse,
    skeleton_hash,
)

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--num_programs", type=int, default=None)
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
ap.add_argument("--grid", type=int, nargs=2, default=None, help="height width")
ap.add_argument("--pairing", default="random", choices=["random", "local"])
ap.add_argument("--frame_every", type=int, default=16)
ap.add_argument("--seed_program", default=None, help="plant this program in the centre")
ap.add_argument("--income", type=int, default=None, help="energy economy income")
a = ap.parse_args()
out = pathlib.Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)

table = opcode_table_swap_heads() if a.dialect == "swap" else None
cs = BFF(
    opcode_table=table,
    mutation_rate=a.mutation_rate,
    heads_from_tape=a.heads,
    control=a.control,
    grid=None if a.grid is None or a.pairing == "random" else tuple(a.grid),
    rngs=nnx.Rngs(a.seed),
)
viewer = (
    BFF(opcode_table=table, grid=tuple(a.grid), rngs=nnx.Rngs(0)) if a.grid else None
)
economy = BFFEconomy(income=a.income, bff=cs) if a.income is not None else None
if a.resume:
    ck = np.load(a.resume)
    soup = jnp.asarray(ck["soup"])
    energy = jnp.asarray(ck["energy"]) if "energy" in ck.files else None
    epoch = int(ck["epoch"])
else:
    soup = cs.init_state(
        num_programs=a.num_programs or (a.grid[0] * a.grid[1] if a.grid else 8192),
        num_soups=a.soups,
    )
    energy = None if economy is None else jnp.full(soup.shape[:-1], a.income, jnp.int32)
    epoch = 0
if soup.ndim == 2:
    soup = soup[None]
    energy = None if energy is None else energy[None]
R = soup.shape[0]
if a.seed_program is not None:
    prog = jnp.asarray(parse(a.seed_program, opcode_table=table))
    centre = 0 if a.grid is None else (a.grid[0] // 2) * a.grid[1] + a.grid[1] // 2
    soup = soup.at[:, centre, : prog.shape[0]].set(prog)


@nnx.jit(static_argnames=("n",))
def chunk(cs, soup, n):
    def body(cs, soup, _):
        keys = jax.random.split(cs.rngs.pairing(), soup.shape[0])
        perm = jax.vmap(lambda k: cs.sample_pairing(k, soup.shape[1]))(keys)
        soup, steps, _ = cs.pair_and_run(soup, perm)
        return soup, steps.mean(axis=-1)

    soup, steps = nnx.scan(
        body,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        length=n,
    )(cs, soup, None)
    return soup, steps.mean(axis=0)  # (soups,)


@nnx.jit(static_argnames=("n",))
def chunk_economy(economy, state, n):
    def body(economy, state, _):
        before = state.energy
        state = economy._step(state)
        deaths = (state.energy == economy.income) & (before != state.energy)
        return state, deaths.mean(axis=-1)

    state, deaths = nnx.scan(
        body,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0),
        out_axes=(nnx.Carry, 0),
        length=n,
    )(economy, state, None)
    return state, deaths.mean(axis=0)


def advance(soup, energy, n):
    """Run `n` epochs; returns (soup, energy, mean steps per pair, death rate)."""
    if economy is None:
        soup, mean_steps = chunk(cs, soup, n)
        return soup, None, mean_steps, jnp.zeros(soup.shape[0])
    state, deaths = chunk_economy(economy, EconomyState(soup=soup, energy=energy), n)
    return state.soup, state.energy, jnp.zeros(soup.shape[0]), deaths


log = open(out / "log.csv", "a")
if log.tell() == 0:
    log.write(
        "epoch,soup,brotli_bytes,h0,hoe,mean_steps,species,largest,energy,deaths,wall_s\n"
    )


def species(s):
    _, counts = np.unique(
        np.asarray(skeleton_hash(s, cs.opcode_table)), return_counts=True
    )
    return len(counts), counts.max() / counts.sum()


if a.grid is not None:
    from PIL import Image

    for i in range(R):
        (out / f"frames_soup{i}").mkdir(exist_ok=True)


def save_frames(soup, epoch):
    if a.grid is None:
        return
    for i in range(R):
        rgb = np.asarray(viewer.render(soup[i]))
        Image.fromarray(rgb).save(out / f"frames_soup{i}" / f"{epoch:09d}.png")


t0 = time.time()
transitioned_at = [None] * R
save_frames(soup, epoch)
while epoch < a.max_epochs:
    for _ in range(a.log_every // a.frame_every):
        soup, energy, mean_steps, deaths = advance(soup, energy, a.frame_every)
        epoch += a.frame_every
        save_frames(soup, epoch)
    all_soups = np.asarray(soup)
    mean_steps = np.asarray(mean_steps)
    for i in range(R):
        s = all_soups[i]
        h0 = byte_entropy(s)
        bpb = compressed_bits_per_byte(s)
        hoe = h0 - bpb
        num_species, largest = species(s)
        mean_energy = 0.0 if energy is None else float(energy[i].mean())
        log.write(
            f"{epoch},{i},{int(bpb * s.size / 8)},{h0:.4f},{hoe:.4f},{float(mean_steps[i]):.1f},{num_species},{largest:.4f},{mean_energy:.1f},{float(deaths[i]):.5f},{time.time() - t0:.1f}\n"
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
        extra = {} if energy is None else {"energy": np.asarray(energy)}
        np.savez_compressed(
            out / f"ck_{epoch:09d}.npz", soup=all_soups, epoch=epoch, **extra
        )
    if all(t is not None and epoch >= t + a.stop_after for t in transitioned_at):
        break
extra = {} if energy is None else {"energy": np.asarray(energy)}
np.savez_compressed(
    out / f"ck_{epoch:09d}_final.npz", soup=np.asarray(soup), epoch=epoch, **extra
)
print("done", epoch, "transition at", transitioned_at)
