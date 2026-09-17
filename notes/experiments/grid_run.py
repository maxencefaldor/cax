"""Run the BFF grid machine and log the memory's complexity signal.

Usage: python grid_run.py OUTDIR --shape H W [--threads N] [--epochs E] [--seed S]
       [--log_every 1] [--frame_every 4] [--mutation_rate 1/4096] [--num_steps 8192]
       [--control flip|cyclic] [--turn reflect|quarter]
An epoch is `num_steps` steps, one thread lifetime. Logs epoch, byte entropy,
high-order entropy, instruction fraction and wall time; saves a frame of `render`
every `frame_every` epochs under `frames/` and a checkpoint every `checkpoint_every`.
"""

import argparse
import pathlib
import time

import jax
import numpy as np
from flax import nnx
from PIL import Image

from cax.cs.bff import (
    BFFGrid,
    GridState,
    byte_entropy,
    compressed_bits_per_byte,
    is_instruction,
    opcode_table_swap_heads,
)

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--shape", type=int, nargs="+", required=True)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--epochs", type=int, default=10**9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log_every", type=int, default=1)
ap.add_argument("--frame_every", type=int, default=4)
ap.add_argument("--checkpoint_every", type=int, default=256)
ap.add_argument("--mutation_rate", type=float, default=1 / 4096)
ap.add_argument("--num_steps", type=int, default=8192)
ap.add_argument("--dialect", default="two", choices=["two", "swap"])
ap.add_argument("--control", default="flip", choices=["flip", "cyclic"])
ap.add_argument(
    "--instruction_frac",
    type=float,
    default=None,
    help="initialise the memory with this fraction of instruction bytes",
)
ap.add_argument("--turn", default="reflect", choices=["quarter", "reflect"])
ap.add_argument("--window", type=int, default=None, help="sandbox side in cells")
ap.add_argument("--anchor", type=int, default=None, help="spawn lattice spacing")
ap.add_argument("--headings", default="both", choices=["both", "positive"])
ap.add_argument("--heads", default="absolute", choices=["absolute", "relative"])
ap.add_argument(
    "--plant",
    default=None,
    help="npy tape planted along the last axis at the centre, with thread 0 on it",
)
ap.add_argument("--resume", default=None)
a = ap.parse_args()
out = pathlib.Path(a.outdir)
(out / "frames").mkdir(parents=True, exist_ok=True)

table = opcode_table_swap_heads() if a.dialect == "swap" else None
cs = BFFGrid(
    opcode_table=table,
    num_steps=a.num_steps,
    mutation_rate=a.mutation_rate,
    control=a.control,
    turn=a.turn,
    window=a.window,
    anchor=a.anchor,
    headings=a.headings,
    heads=a.heads,
    rngs=nnx.Rngs(a.seed),
)
if a.resume:
    ck = np.load(a.resume)
    state = GridState(**{k: jax.numpy.asarray(ck[k]) for k in ck.files if k != "epoch"})
    epoch = int(ck["epoch"])
else:
    state = cs.init_state(shape=tuple(a.shape), num_threads=a.threads)
    epoch = 0
    if a.instruction_frac is not None:
        rng = np.random.default_rng(a.seed)
        instr = np.asarray(is_instruction(cs.opcode_table))
        bytes_ = np.arange(256, dtype=np.uint8)
        coin = rng.random(tuple(a.shape)) < a.instruction_frac
        memory = np.where(
            coin,
            rng.choice(bytes_[instr], tuple(a.shape)),
            rng.choice(bytes_[~instr], tuple(a.shape)),
        ).astype(np.uint8)
        state = GridState(**{**vars(state), "memory": jax.numpy.asarray(memory)})
    if a.plant is not None:
        tape = np.load(a.plant)
        memory = np.array(state.memory)
        centre = tuple(side // 2 for side in a.shape)
        start = centre[:-1] + (centre[-1] - (a.window or 2 * tape.shape[0]) // 2,)
        memory[start[:-1] + (slice(start[-1], centre[-1]),)] = tape
        # Thread 0 starts at the tape's first byte heading along the last axis, with
        # its window centred where a BFF pair's second half would begin.
        fields = {k: np.array(v) for k, v in vars(state).items()}
        for k in ("position", "head0", "head1"):
            fields[k][0] = start
        fields["origin"][0] = centre
        fields["direction"][0] = len(a.shape) - 1
        fields["age"][0] = 0
        fields["memory"] = memory
        state = GridState(**{k: jax.numpy.asarray(v) for k, v in fields.items()})


def save_frame(state: GridState, epoch: int) -> None:
    Image.fromarray(np.asarray(cs.render(state))).save(
        out / "frames" / f"{epoch:07d}.png"
    )


def checkpoint(state: GridState, epoch: int, tag: str = "") -> None:
    arrays = {k: np.asarray(v) for k, v in vars(state).items()}
    np.savez_compressed(out / f"ck_{epoch:07d}{tag}.npz", epoch=epoch, **arrays)


log = (out / "log.csv").open("a")
if log.tell() == 0:
    log.write("epoch,h0,hoe,instruction_frac,wall_s\n")
t0 = time.time()
save_frame(state, epoch)
while epoch < a.epochs:
    state = cs(state, num_steps=a.num_steps)
    epoch += 1
    if epoch % a.frame_every == 0:
        save_frame(state, epoch)
    if epoch % a.log_every == 0:
        memory = np.asarray(state.memory)
        h0 = byte_entropy(memory)
        hoe = h0 - compressed_bits_per_byte(memory)
        frac = float(np.mean(np.asarray(is_instruction(cs.opcode_table))[memory]))
        log.write(f"{epoch},{h0:.4f},{hoe:.4f},{frac:.4f},{time.time() - t0:.1f}\n")
        log.flush()
    if epoch % a.checkpoint_every == 0:
        checkpoint(state, epoch)
checkpoint(state, epoch, "_final")
print("done", epoch)
