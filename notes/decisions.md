# Decisions log

Newest first.
Each entry: what was decided, the evidence, what it would take to reverse.

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
