"""One attempt of the kernel hill-climb: exactness gate, then timing on four regimes.

Usage: python kernel_attempts.py "description" [--kw block=32 ...] [--no-log]
Regimes (65,536 pairs each, 8192 steps, `bff_noheads`, ASCII table):
  random      uniform bytes (most tapes halt within ~600 steps)
  enriched    50% instruction bytes, 10% zeros (long-running, many jumps)
  mid         soup at the transition epoch of phase-1 seed 0 (hoe 2.6)
  final       the same soup 4,096 epochs later (hoe 5.3, replicators everywhere)
Exactness: kernel vs the XLA scan on 2,048 tapes of each regime for all three control
flows, both head conventions and both opcode tables; any mismatch fails the attempt.
Appends a row to results/kernel_attempts.csv with the UTC time, so attempts plot over
time; cubff on the same box does random in 10.3 ms and final in 35 ms per epoch.
"""

import argparse
import csv
import datetime
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import opcode_table_from_string, opcode_table_swap_heads, run
from cax.cs.bff.kernel import run_kernel

HERE = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("description")
ap.add_argument("--kw", nargs="*", default=[], help="kernel kwargs as key=value")
ap.add_argument("--no-log", action="store_true")
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--bench_dir", default="runs/bench")
a = ap.parse_args()
kw = {}
for item in a.kw:
    k, v = item.split("=")
    try:
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        kw[k] = v == "True" if v in ("True", "False") else v

bench = pathlib.Path(a.bench_dir)
bench.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def load_or_make(name, make):
    f = bench / f"{name}.npy"
    if not f.exists():
        np.save(f, make())
    return np.load(f)


def pairs_of(soup):
    perm = np.random.default_rng(1).permutation(soup.shape[0])
    return soup[perm].reshape(-1, 128)


def enriched(n):
    ops = np.frombuffer(b"[]+-.,<>{}", dtype=np.uint8)
    t = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    m = rng.random(t.shape)
    t = np.where(m < 0.4, rng.choice(ops, t.shape), t)
    return np.where((m >= 0.4) & (m < 0.5), 0, t).astype(np.uint8)


workloads = {
    "random": load_or_make(
        "random", lambda: pairs_of(rng.integers(0, 256, (131072, 64), dtype=np.uint8))
    ),
    "enriched": load_or_make("enriched", lambda: pairs_of(enriched(131072))),
    "mid": load_or_make(
        "mid",
        lambda: pairs_of(np.load("runs/phase1/s0/ck_000002176_transition.npz")["soup"]),
    ),
    "final": load_or_make(
        "final",
        lambda: pairs_of(np.load("runs/phase1/s0/ck_000006272_final.npz")["soup"]),
    ),
}

# Exactness gate.
tables = {"ascii": opcode_table_from_string(), "swap": opcode_table_swap_heads()}
failures = 0
for tname, table in tables.items():
    for control in ("matched", "cyclic", "flip"):
        for heads in (False, True):
            steps = 8192 if control == "matched" else 1024
            for wname, w in workloads.items():
                tapes = jnp.asarray(w[:2048])
                ref = run(
                    tapes,
                    table,
                    num_steps=steps,
                    heads_from_tape=heads,
                    control=control,
                    implementation="xla",
                )
                out = run_kernel(
                    tapes,
                    table,
                    num_steps=steps,
                    heads_from_tape=heads,
                    control=control,
                    **kw,
                )
                bad = [int((x != y).sum()) for x, y in zip(out, ref)]
                if any(bad):
                    failures += 1
                    print(
                        f"MISMATCH {tname} {control} heads={heads} {wname}: {bad}",
                        flush=True,
                    )
print("exactness failures:", failures, flush=True)

# Timing.
table = tables["ascii"]
times = {}
for wname, w in workloads.items():
    tapes = jnp.asarray(w)
    f = jax.jit(lambda t: run_kernel(t, table, num_steps=8192, **kw))
    f(tapes)[0].block_until_ready()
    ts = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        f(tapes)[0].block_until_ready()
        ts.append(time.perf_counter() - t0)
    times[wname] = 1e3 * float(np.median(ts))
    print(
        f"{wname:9s}: {times[wname]:8.2f} ms   (mean steps {float(f(tapes)[1].mean()):6.0f})",
        flush=True,
    )
score = float(np.mean(list(times.values())))
print(f"mean over regimes: {score:.2f} ms", flush=True)

# Eight random soups stacked into one launch, per soup: the many-seed production case.
stacked = jnp.asarray(
    np.concatenate([np.roll(workloads["random"], 17 * i, axis=0) for i in range(8)])
)
f = jax.jit(lambda t: run_kernel(t, table, num_steps=8192, **kw))
f(stacked)[0].block_until_ready()
ts = []
for _ in range(a.repeats):
    t0 = time.perf_counter()
    f(stacked)[0].block_until_ready()
    ts.append(time.perf_counter() - t0)
batch8 = 1e3 * float(np.median(ts)) / 8
print(f"random, 8 soups per launch: {batch8:8.2f} ms per soup", flush=True)

# The total machines of Phase 2 never halt: every lane runs the whole budget and the
# search differs (`cyclic` walks a ring, `flip` has none). Timed on the enriched regime.
extra = {}
for control in ("cyclic", "flip"):
    tapes = jnp.asarray(workloads["enriched"])
    f = jax.jit(
        lambda t, c=control: run_kernel(t, table, num_steps=8192, control=c, **kw)
    )
    f(tapes)[0].block_until_ready()
    ts = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        f(tapes)[0].block_until_ready()
        ts.append(time.perf_counter() - t0)
    extra[control] = 1e3 * float(np.median(ts))
    print(f"{control:9s} on enriched: {extra[control]:8.2f} ms", flush=True)

if not a.no_log:
    log = HERE / "results" / "kernel_attempts.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(
                [
                    "utc",
                    "description",
                    "kwargs",
                    "exact",
                    *workloads,
                    "mean_ms",
                    "cyclic_ms",
                    "flip_ms",
                ]
            )
        wr.writerow(
            [
                datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                a.description,
                " ".join(a.kw),
                failures == 0,
                *[f"{times[w]:.2f}" for w in workloads],
                f"{score:.2f}",
                f"{extra['cyclic']:.2f}",
                f"{extra['flip']:.2f}",
            ]
        )
        wr.writerow(
            [
                datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                a.description,
                " ".join(a.kw),
                failures == 0,
                *[f"{times[w]:.2f}" for w in workloads],
                f"{score:.2f}",
            ]
        )
