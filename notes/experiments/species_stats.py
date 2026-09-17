"""Species statistics of saved soups.

Usage: species_stats.py CKPT.npz [CKPT.npz ...]
Two species definitions: the positional skeleton (`skeleton_hash`: same instructions
at the same positions) and the instruction sequence (the same instructions in the same
order, wherever they sit, so shifted and data-mutated copies are one species). For each,
per soup: number of species, effective number (exp of the Shannon entropy of shares),
number holding at least 0.1% of the soup, and the share of the largest.
"""

import sys

import numpy as np

from cax.cs.bff import Op, is_instruction, opcode_table_from_string

TABLE = np.asarray(opcode_table_from_string())
INSTRUCTION = np.asarray(is_instruction(TABLE))


def hashes(soup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    op = TABLE[soup].astype(np.uint64)
    mask = INSTRUCTION[soup]
    weights = np.cumprod(np.full(soup.shape[-1], 1000003, dtype=np.uint64))
    skeleton = (np.where(mask, op + 1, 0) * weights).sum(axis=-1)
    rank = np.cumsum(mask, axis=-1) - 1
    sequence = np.where(mask, (op + 1) * weights[np.maximum(rank, 0)], 0).sum(axis=-1)
    return skeleton, sequence


def stats(h: np.ndarray) -> str:
    _, counts = np.unique(h, return_counts=True)
    share = counts / counts.sum()
    effective = float(np.exp(-(share * np.log(share)).sum()))
    return (
        f"{len(counts)},{effective:.1f},{int((share >= 1e-3).sum())},{share.max():.4f}"
    )


print(
    "checkpoint,soup,epoch,skeleton_species,skeleton_effective,skeleton_abundant,"
    "skeleton_largest,sequence_species,sequence_effective,sequence_abundant,"
    "sequence_largest"
)
for path in sys.argv[1:]:
    ck = np.load(path)
    soups = ck["soup"]
    soups = soups[None] if soups.ndim == 2 else soups
    for i, soup in enumerate(soups):
        skeleton, sequence = hashes(soup)
        print(f"{path},{i},{int(ck['epoch'])},{stats(skeleton)},{stats(sequence)}")
