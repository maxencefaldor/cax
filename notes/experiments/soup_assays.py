"""Behaviour vectors, cores and robustness of a soup's lineages, over checkpoints.

Usage: soup_assays.py OUT.csv CKPT.npz [CKPT.npz ...] [--control matched] [--top 8]
       [--soup 0] [--scan_top 1]
For every checkpoint (all soups in it unless --soup is given): the most common exact
tapes, each with the share of the instruction-sequence lineage it belongs to, get the
seven assays (partners drawn from the soup and at random, host = the dominant tape);
the top `scan_top` also get the mutational scan (core length, core span, robustness).
One CSV row per lineage per checkpoint; the dominant lineage's hash per checkpoint
gives the turnover.
"""

import argparse
import csv
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import (
    NUM_CHAINS,
    assays,
    mutational_scan,
    opcode_table_from_string,
    sample_partners,
    unparse,
)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--control", default="matched", choices=["matched", "cyclic", "flip"])
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--soup", type=int, default=None)
ap.add_argument("--scan_top", type=int, default=1)
a = ap.parse_args()

TABLE = opcode_table_from_string()
NP_TABLE = np.asarray(TABLE)
INSTRUCTION = NP_TABLE < 10


def sequence_hash(soup: np.ndarray) -> np.ndarray:
    """Hash of each tape's instruction sequence, ignoring where the instructions sit."""
    op = NP_TABLE[soup].astype(np.uint64)
    mask = INSTRUCTION[soup]
    weights = np.cumprod(np.full(soup.shape[-1], 1000003, dtype=np.uint64))
    rank = np.cumsum(mask, axis=-1) - 1
    return np.where(mask, (op + 1) * weights[np.maximum(rank, 0)], 0).sum(axis=-1)


def lineages(soup: np.ndarray, top: int) -> list[tuple[int, float, np.ndarray]]:
    """(lineage hash, lineage share, tape) for the `top` most common exact tapes.

    The most common exact tapes are the canonical forms of the soup's replicators;
    the lineage is the instruction sequence each belongs to, with its share.
    """
    h = sequence_hash(soup)
    values, counts = np.unique(h, return_counts=True)
    share = dict(zip(values.tolist(), (counts / len(soup)).tolist(), strict=True))
    tapes, n = np.unique(soup, axis=0, return_counts=True)
    out = []
    for i in np.argsort(-n)[:top]:
        lineage = int(sequence_hash(tapes[i][None])[0])
        out.append((lineage, share[lineage], tapes[i]))
    return out


fields = [
    "checkpoint",
    "soup",
    "epoch",
    "rank",
    "lineage",
    "share",
    "program",
    "replicates",
    "replicates_in_soup",
    "replicates_with_kin",
    "survives",
    "survives_in_soup",
    "overwrites_host",
    "copies_partner",
    "spends",
    "partner_pays",
    "core_length",
    "core_span",
    "robustness",
]
path = pathlib.Path(a.out)
new = not path.exists()
with path.open("a") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for ck_path in a.ckpts:
        ck = np.load(ck_path)
        soups = ck["soup"]
        soups = soups[None] if soups.ndim == 2 else soups
        epoch = int(ck["epoch"])
        for si, soup in enumerate(soups):
            if a.soup is not None and si != a.soup:
                continue
            tops = lineages(soup, a.top)
            programs = jnp.asarray(np.stack([t[2] for t in tops]))
            num = programs.shape[0]
            rng = np.random.default_rng(epoch + si)
            idx = rng.integers(0, len(soup), (num, NUM_CHAINS))
            soup_partners = jnp.asarray(soup[idx])
            out = assays(
                programs,
                sample_partners(jax.random.key(epoch + si), num),
                soup_partners,
                programs[0],
                TABLE,
                control=a.control,
            )
            out = {k: np.asarray(v) for k, v in out.items()}
            for rank, (h, share, rep) in enumerate(tops):
                row = {
                    "checkpoint": ck_path,
                    "soup": si,
                    "epoch": epoch,
                    "rank": rank,
                    "lineage": h,
                    "share": f"{share:.4f}",
                    "program": unparse(rep),
                    **{k: f"{float(v[rank]):.3f}" for k, v in out.items()},
                    "core_length": "",
                    "core_span": "",
                    "robustness": "",
                }
                if rank < a.scan_top:
                    scan = np.asarray(
                        mutational_scan(
                            jnp.asarray(rep),
                            jax.random.key(rank),
                            TABLE,
                            control=a.control,
                            threshold=3 * soup.shape[1] // 4,
                        )
                    )
                    core = np.nonzero(scan < 0.5)[0]
                    row["core_length"] = len(core)
                    row["core_span"] = (
                        int(core.max() - core.min() + 1) if len(core) else 0
                    )
                    row["robustness"] = f"{scan.mean():.3f}"
                w.writerow(row)
                f.flush()
            print(ck_path, si, epoch, "top", tops[0][1], "done", flush=True)
