# Phase 2 design: principled BFF (draft, 2026-09-16)

Phase 1 is reproduced as far as this machine allows (exactness, seeded takeover, emergence via the reference at reduced size, CAX tracking the reference's own takeover).
Phase 2 starts now.

## What "principled" is measured against

The baseline is `bff_noheads` (Phase 1 default).
Every variant is a *parameter* of the same interpreter, so the soup, detector and metrics are shared and only the step function changes.

## The two axes to vary

1. **Control flow** (the only non-local, non-total rule in BFF):
   - `matched` (baseline): scan to the matching bracket, halt if none.
   - `flip`: the thread carries a direction; `]` reverses it when the byte at head0 is nonzero, `[` when zero.
     No scan, no halt rule.
     O(1).
   - `cyclic`: circular tape, jump to the nearest bracket cyclically.
     No halt rule, still O(N).
1. **Heads**: `two` (baseline `< > { } . ,`) or `swap` (`< > . ~`, where `~` exchanges head0 and head1; `,` is `~.~`).

Halting: with `matched`, falling off the end and unmatched brackets halt.
With `flip` and `cyclic` the tape is cyclic and only the step budget ends a run: totality.

## The first experiment, and why it is first

The 2026 paper's metric: **programs tested per replicator found**, sampling 64-byte tapes from a stated byte distribution and scoring them with the detector.
No soup, no takeover dynamics, one number per variant, and it is exactly the quantity the brief's "shortest replicator / emergence probability" argument is about.
Two distributions: uniform bytes, and CUST (50% instruction bytes).
Budget: 2×10^5 programs per variant per distribution at first (about 10^7 executions), scaled up where the count is small.

Secondary measurements on the same samples, all cheap:

- fraction of tapes that write at least one byte (the total-machine replacement for "halts");
- mean steps executed (cost profile of each variant);
- length of the shortest replicator found (bytes between the first and last instruction it executes), by exhaustive search over short programs where feasible (all 256^k for k ≤ 3 is 1.7×10^7; k = 4 needs sampling).

Then, for the variants that produce replicators at a comparable or better rate: seeded takeover and emergence in the soup, as in Phase 1.

## Predictions to be wrong about

- `flip` makes replicators *shorter* (a bounce loop needs one bracket pair and no zero cell to terminate on) but makes copy loops copy every byte twice per bounce, so the minimal *working* copier may be the same length.
- `swap` costs about one byte per head move, so its replicators are longer and rarer by a factor near 256; the 2026 paper's `bff_selfmove` (one copy op, auto-advance) suggests the count of replicators matters more than the minimal length.
- Totality removes the cheap "death" of random tapes, so the soup's compute per epoch rises ~10× and the survivor compaction stops paying; this is the cost of the principled design and must be reported with the result.

## Implemented (2026-09-16, evening)

- `run(..., control="matched" | "cyclic" | "flip")`, `opcode_table_swap_heads()` for the `~` dialect; detector and `BFF` take `control`.
- `flip` semantics as built: brackets are *mirrors on both faces* (`[` reflects when the test byte is zero, `]` when nonzero), so a loop is a room `] body ]` or `[ body [`: the pointer bounces inside while the wall condition holds and leaves through whichever wall opens.
  A pointer outside can only enter while the wall is open, so the idiom "enter when zero, loop while nonzero" is `] + body ]`.
  This differs from the brief's sketch (`[.>}]` style loops); the one-way-mirror alternative (reflect only when approached from inside) was considered and rejected because it makes an exit run the preceding code backwards.
- Exactness of `matched` re-verified against cubff after the refactor (0 mismatches on all three checks).
- First experiment running: `appearance_rate.py`, 20k programs per (control × heads × distribution), 12 combinations.

## Result: programs tested per replicator found (2026-09-17, night)

`appearance_rate.py`, 10^7 programs per combination, 64-byte tapes, the detector's score ≥ 60 as "replicator" (`runs/phase2/appearance_10m.csv`; the 2 × 10^5 pilot is `appearance_200k.csv`).
Uniform bytes, and "cust": half the bytes instructions.

| control | heads | distribution | replicators per 10^7 | programs per replicator | writes anything | mean steps |
| --- | --- | --- | --- | --- | --- | --- |
| matched | two | uniform | 0 (best score 1) | > 10^7 | 59 % | 615 |
| matched | two | cust | 10 | 1.0 × 10^6 | 72 % | 2,391 |
| matched | swap | uniform | 0 | > 10^7 | 51 % | 636 |
| matched | swap | cust | 0 (best 1) | > 10^7 | 67 % | 2,699 |
| cyclic | two | uniform | 0 (one at 32) | > 10^7 | 75 % | 8,192 |
| cyclic | two | cust | 18 | 5.6 × 10^5 | 90 % | 8,192 |
| cyclic | swap | any | 0 | > 10^7 | | 8,192 |
| flip | two | uniform | 0 (best 2) | > 10^7 | 77 % | 8,192 |
| flip | two | cust | 0 (best 4) | > 10^7 | 95 % | 8,192 |
| flip | swap | any | 0 | > 10^7 | | 8,192 |

The 2026 paper's number for `bff_noheads` under its instruction-enriched distribution is 1.7 × 10^6; ours under a plainer 50 % mix is 1.0 × 10^6, the same order.
Under uniform bytes the paper reports 2.9 × 10^7 and we expect about 0.3 hits in 10^7, so zero is consistent.

What it says:

- **Direction-flip brackets produce no replicators at all** at a budget where matched brackets produce ten.
  The best score in 2 × 10^7 flip programs is 4 of 64.
  The brief's favoured primitive fails the first test; a bounce loop copies every byte twice per cycle and cannot skip, and apparently no short program works around that.
- **Cyclic nearest-bracket is at least as good as matched**: 18 against 10, no halt rule, a ring tape, totality.
  It costs the O(N) search, which the kernel already pays at 11.5 ms per 65k pairs.
  This is the principled variant to build on.
- **Swap heads are at least ten times rarer** than two heads under every control flow: zero in 10^7 against ten.
  The brief's "256² rarer" guess is not testable at this budget, but the direction is confirmed and the swap dialect is dropped.
- Totality shows in the cost column: every `cyclic` and `flip` program runs the full 8,192 steps.

Consequence for the grid (`09_grid_design.md`): the grid machine was built on flip.
It now has `control="cyclic"`, where brackets jump along the thread's line with nesting and wrap, so its 1D instance is the cyclic machine exactly (`test_grid_1d_matches_interpreter`), and in 2D it is a weave of row and column machines sharing every cell.
Runs with both controls are in `runs/grid3`.

## Result: cyclic soups transition in every seed (2026-09-17, 02:00)

Four soups of 2^17 programs under `control="cyclic"`, random pairing, mutation 1/4096, 32,768 epochs (`runs/phase2/cyclic_soups`, seeds 40-43, 16 minutes on one GPU for all four).
High-order entropy is not a usable transition signal for a total machine: it reaches 3.6 by epoch 64 because bounce and wrap loops smear single bytes into long runs before any replicator exists.
The detector is the signal instead (`replication_score`, 1,024 sampled tapes per soup and checkpoint, score ≥ 60 of 64):

| epoch | soup 0 | soup 1 | soup 2 | soup 3 |
| --- | --- | --- | --- | --- |
| 8,192 | 94 % | 0 % | 85 % | 0 % |
| 16,384 | 95 % | 68 % | 90 % | 92 % |
| 32,768 | 96 % | 67 % | 95 % | 97 % |

Four of four soups are replicator-dominated by 16k epochs; under `matched` the fraction is 37 % of 1,008 seeds (`06`).
The replicators score 64 under `matched` too: the winners are ordinary BFF replicators, palindromic tapes such as `[<,} ] },<[` with their mirror images, that the ring finds faster.
Four seeds; the statistic needs the same 1,000-seed treatment as Phase 1, at 11 ms per epoch with every tape alive.

So the principled variant that removes the halt rule and makes the tape a ring is not a cost: it finds replicators sooner in the detector and in the soup.
Totality is paid for in compute (every tape runs the full budget, 2.5× the matched epoch) and in the loss of the compression signal before takeover.

## The cyclic statistic, first batches (05:25 London)

`runs/phase2/cyclic_stat`, batches of 8 soups on four GPUs from 05:05, scored by `cyclic_stat_score.py` (512 sampled tapes per soup, a soup counts when half of them score ≥ 60):

| epoch | soups replicator-dominated |
| --- | --- |
| 8,192 | 55 of 96 (57 %) |
| 16,384 | 80 of 96 (83 %, ± 7.5) |

The reference machine's figure is 38 % of 1,008 (`06`); the ring more than doubles it.
More batches land through the morning; the launcher runs 384 soups in all.
