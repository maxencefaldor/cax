# Decisions log

Newest first.
Each entry: what was decided, the evidence, what it would take to reverse.

## 2026-09-16 · The GPU interpreter is one Pallas kernel; the XLA scan stays as the portable reference

Measured (`05_gpu_throughput.md`): the XLA scan's step costs 40 µs on an H100 at any batch size because it is 20 to 40 kernels and nothing (unrolling, CUDA graphs, a different prefix sum) fuses them.
`kernel.run_kernel` runs a warp of 32 tapes for the whole budget inside one kernel with real halting: 19 ms for 65k pairs against 449 ms, exact against the scan on 24 configurations.
`run(implementation="auto")` picks it on GPU.
The scan remains the CPU/TPU path and the readable reference; both are tested against `step`.
Reverse if a JAX release makes the scan competitive, or if Pallas Triton drops a primitive the kernel uses.

## 2026-09-16 · Sharding is a `BFF` option, not the default

`BFF(shard_axis=name)` shards the soup over a mesh axis: gather and mutation under the partitioner, the run under `jax.shard_map`, write-back as a gather through the inverse permutation.
Bit-identical to the unsharded epoch.
Kept off by default because one H100 handles the paper's soup in 21 ms and the statistics the programme needs are many independent soups; it exists for soups of millions of programs.

## 2026-09-16 · One process per GPU; no vmap over soups; shard only soups ≥ 2^22

Measured on H100 (`05_gpu_throughput.md`): the epoch cost is a ~40 µs per-step floor times 8192 steps, independent of batch size up to ~2^18 pairs.
Four processes sharing a GPU gain nothing; two soups under `nnx.vmap` cost 4× per run-epoch because `lax.cond` becomes `select` and the compaction fallback runs every epoch.
Sharding one soup over devices (`jax.make_mesh` + `device_put(P("program"))`, no other code) pays only when one device is compute-bound, i.e. soups of 2^22 programs and up.
Reverse if the step floor is cut by an order of magnitude, which would make the batch the limiting factor again.

## 2026-09-16 · Exactness re-verified on the GPU build

Same three checks as the CPU acceptance test, JAX on H100, cubff rebuilt here for `sm_90`: 0 mismatches.
The fidelity scripts take `CUBFF_BIN` and `CUBFF_REPLAY` from the environment.

## 2026-09-16 · Phase 1 is `bff_noheads`; head initialisation is a flag

The paper's Section 2 language starts IP and heads at 0 and the paper says so explicitly (`--lang bff_noheads`).
cubff's `bff` seeds heads from the first two bytes.
Both are one boolean apart, so `BFF(heads_from_tape=...)` covers both; the default is the paper's.
Reverse if the user wants the Python tooling's default (`bff`) as the baseline instead.

## 2026-09-16 · Byte-exactness against cubff is the acceptance test for the interpreter

Three checks, all exact: 16k random tapes through `Bff::Evaluate` (both head conventions), 4 whole epochs of a 1024-program soup with cubff's own RNG replayed, and the replication detector on 204 programs with cubff's own noise.
Any semantic change in later phases must be a *new* system or flag; this one stays exact.
Scripts in `experiments/`.

## 2026-09-16 · Fast path = on-demand bracket search + survivor compaction, with exact fallback

Measured: 14 ns per tape-step without bracket matching, 150–225 ns with it.
The batched step searches only for tapes taking a jump (0.6% of a random soup per step, 9% of the survivors), gathered into a fixed buffer; overflow falls back to searching everything.
Survivors are compacted once after 256 steps (8% survive).
Result is identical either way; a test runs the fast path against the single-thread `step` under three buffer settings.
Full-size epoch: 3.7 s vs cubff's 0.13 s single-threaded.
Defaults chosen by benchmark (`scan_fraction=1/16`, `compact_after=256`, `compact_fraction=1/8`).

## 2026-09-16 · The state is the soup array, not a dataclass

Nothing else survives an epoch: pairing and mutation are drawn fresh from named RNG streams (`pairing`, `mutation`), the initial soup from `params`.
Per-epoch diagnostics (steps per pair) are returned by `pair_and_run`, not carried.
Phase 3 will need a dataclass (grid + threads); that is a different system.

## 2026-09-16 · Complexity metric is brotli-based high-order entropy, computed host-side

The paper uses brotli q2 / lgwin 24 and reports byte entropy minus compressed bits per byte.
`brotli` is an optional dependency (examples extra); zlib is offered as a fallback with different absolute values.
Threshold for "transitioned": report both 1 (paper Fig. 6) and 3 (cubff scripts); in practice the signal is bimodal and both agree.

## 2026-09-16 · Reproduction strategy given CPU-only compute

Full-scale (1000 × 16k epochs × 2^17) is out of reach on a laptop for either implementation.
Plan: (a) exactness (done); (b) the paper's *seeded* experiment (one hand-written replicator, 128 epochs, ~22% takeover at 2^17) at reduced and full size in cubff and at reduced size in CAX, compared as rates; (c) emergence from random soups at N = 8192 in both cubff (fast, many seeds) and CAX (slow, few seeds), compared as time-to-transition in *interactions*.
cubff built locally is the ground truth at scale.

## 2026-09-16 · Detector threshold

Ported `CheckSelfRep` exactly (13 chains × 5 executions, agreement in > 3 chains, min over halves).
The paper's hand-written replicator scores 17 (its own length); evolved replicators that copy the whole tape score near 64.
The 2026 paper's threshold 48 therefore excludes short replicators; cubff's display threshold is 5.
The score is exposed and the threshold is the caller's; experiments report both.
