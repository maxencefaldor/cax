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
