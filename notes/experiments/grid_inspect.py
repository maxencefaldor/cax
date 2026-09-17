"""Inspect a grid-machine checkpoint: the most common byte windows in the memory.

Usage: grid_inspect.py CKPT.npz [--window 16] [--top 12]
Prints byte entropy, high-order entropy, instruction fraction, and the most frequent
`window`-byte strings along the last axis, unparsed, with their counts.
"""

import argparse

import numpy as np

from cax.cs.bff import (
    byte_entropy,
    high_order_entropy,
    is_instruction,
    opcode_table_from_string,
    unparse,
)

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--window", type=int, default=16)
ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()

ck = np.load(a.ckpt)
memory = ck["memory"]
flat = memory.reshape(-1, memory.shape[-1]) if memory.ndim > 1 else memory[None]
table = np.asarray(opcode_table_from_string())
frac = float(np.mean(np.asarray(is_instruction(table))[memory]))
print(
    f"epoch {int(ck['epoch'])} shape {memory.shape} h0 {byte_entropy(memory):.3f} "
    f"hoe {high_order_entropy(memory):.3f} instruction fraction {frac:.3f}"
)
windows = np.lib.stride_tricks.sliding_window_view(flat, a.window, axis=-1).reshape(
    -1, a.window
)
keys = np.ascontiguousarray(windows).view(f"S{a.window}").ravel()
values, counts = np.unique(keys, return_counts=True)
order = np.argsort(-counts)[: a.top]
for i in order:
    if counts[i] < 2:
        break
    print(f"{counts[i]:8d}  {unparse(np.frombuffer(values[i], dtype=np.uint8))}")
