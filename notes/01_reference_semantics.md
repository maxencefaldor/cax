# BFF reference semantics, verified against cubff (commit f212e84)

Source of truth: `bff.inc.h` (`Bff::EvaluateOne`, `Bff::Evaluate`, `InitialState`) and `common_language.h` (`InitPrograms`, `MutateAndRunPrograms`, `RunSimulation`, `CheckSelfRep`).
Everything here is what the code does, not what the paper says; the two differ in one place (head initialisation, see §3 of `00_critique.md`).

## Machine

- Tape: 128 bytes = two 64-byte programs concatenated.
  Bytes are unsigned and wrap.
- Registers: `pc`, `head0`, `head1` (int).
  Heads are masked `& 127` at the start of each step; the step then reads and writes with the masked value.
  Equivalent to wrapping on move.
- Opcode of a byte: `[ ] + - . , < > { }` at their ASCII codes (91 93 43 45 46 44 60 62 123 125); byte 0 is `kNull`; everything else `kNoop`.
  `bff_perm` packs the ten instructions into bytes 1..10 (`< > { } + - . , [ ]`); `bff_selfmove` (2026 paper) uses bytes 0..6 for `> < + - , [ ]`, has no `.`, `{`, `}`, and its `,` also does `head1++`.
- Per step, in this order: decode `tape[pc]`; execute; then `if pc < 0: halt`; `pc += 1`; `if pc >= 128: halt`.
  The step counts toward the budget whether or not the byte was an instruction.
  `Evaluate` returns `steps - noops` ("ops").
- `+`/`-`: `tape[head0] ± 1`.
  `.`: `tape[head1] = tape[head0]`.
  `,`: `tape[head0] = tape[head1]`.
  `<`/`>`: `head0 ∓ 1`.
  `{`/`}`: `head1 ∓ 1`.
- `[`: if `tape[head0] == 0` (the raw byte, so `kNull`; any nonzero byte is "true"), scan forward from `pc+1` counting `[` as +1 and `]` as −1 from 1 until 0; land on the matching `]` (then the common `pc += 1` moves past it).
  If the scan reaches the end without matching: `pc = 128`, halt.
  If `tape[head0] != 0`: nothing, no scan.
- `]`: mirror image.
  If `tape[head0] != 0`, scan backward; land on the matching `[` (then `pc += 1` re-enters the loop body).
  Unmatched: `pc = -1`, halt.
- Step budget: 8192 (`Evaluate(tape, 8 * 1024)`), used both in the soup and in the
  replication detector.
- Initial state, `bff_noheads` (`#ifndef BFF_HEADS`): `head0 = head1 = 128 & 127 = 0`, `pc = 0`.
  `bff` (`BFF_HEADS`): `head0 = tape[0] % 128`, `head1 = tape[1] % 128`, `pc = 2`.

## Soup

- `num_programs` = 2^17 by default, 64 bytes each, initialised i.i.d. uniform over bytes
  via `SplitMix64(64 * N * seed0 + 64 * i + j) % 256`.
- Every epoch: Fisher-Yates permutation of all indices (from the top, `j = SplitMix64( seed(epoch * N + i)) % (i + 1)`); pair `k` is `(perm[2k], perm[2k+1])`, first index in bytes 0..63.
  Every program is in exactly one pair per epoch.
- Mutation before execution, per byte of the 128-byte pair tape: with probability `mutation_prob / 2^30` replace by a uniform random byte; default `mutation_prob = 2^18`, i.e. 1/4096 = 0.0244%.
  `--mutation_prob` in `main.cc` defaults to 1/(256·16), the same value.
- Seeds: `seed(x) = SplitMix64(SplitMix64(params.seed) ^ SplitMix64(x))`.
- Reported every `callback_interval` epochs: brotli (quality 2, lgwin 24, generic) size of the whole soup; `h0` = byte Shannon entropy; `higher_entropy = h0 − brotli_bits/byte`.
  The paper's Fig. 6 threshold for "transitioned" is 1; the scripts `time_to_sr.py` and `runit.py` use 3.
- `kSelfrepThreshold = 5` is the score at which `main.cc` counts a program as a replicator in its display; the 2026 paper uses 48.
  The detector (`CheckSelfRep`): 13 chains; each chain pairs the program with one fixed random 64-byte partner, runs, then four more times moves the second half to the first and re-pairs with the *same* partner bytes, runs.
  Score per half = number of byte positions where some chain's value (which for the first half must also equal the original program's byte) is shared by more than 13/4 = 3 chains; score = min(first half, second half).

## Fidelity checks performed (all exact)

1. `experiments/fidelity.py` + `experiments/harness.cc`: 4 × 4096 tapes (uniform and
   instruction-enriched), both head conventions, 8192 steps: 0 tape mismatches, 0 ops
   mismatches against `Bff::Evaluate`.
1. `experiments/replay.py`: cubff's `SplitMix64` initialisation, permutation and
   mutation replayed host-side; `BFF.pair_and_run` reproduces cubff's per-epoch
   checkpoints for `bff_noheads` and `bff` (N = 1024, 4 epochs, seed 7): 0 mismatching
   bytes.
1. cubff itself, built here for CPU without OpenMP, reproduces its shipped
   `testdata/bff.txt` log (seed 10248, 256 epochs) with brotli 1.2.0.

## Performance model (this machine, Apple Silicon, XLA CPU)

- Per tape-step without bracket search: 14 ns.
  With a full 128-wide search every step: 150–225 ns.
  Hence `run` searches only for threads that take a jump (gathered into a fixed buffer, full search as fallback) and compacts survivors after 256 steps.
- Random soup: 8% of pairs survive 256 steps; 0.6% of tapes (9% of survivors) take a
  jump on a given step.
- cubff single-threaded: 0.13 s per 2^17-program epoch on a random soup (2 µs per pair).
- A JAX process on this machine runs 50 threads regardless of `XLA_FLAGS`; two concurrent CAX runs plus four cubff runs drove the load average to 50 and slowed every job several-fold.
  Run at most two CAX processes at a time.
