# Night report, 2026-09-16 → 17

Everything below ran between 23:00 and 07:00 London time on the eight H100s, from the state at commit `f3c2400` (kernel v1).
Report page: <https://claude.ai/artifact/Gq33YYAucvWBQcw82v43QZ>.
Pictures and videos: <https://claude.ai/artifact/LH9URpWyHwb6Qi8q32LCvX>.
Details per topic: `03` (Phase 2), `06` and `runs/phase1` (Phase 1), `08` (space), `09` (grid machine).

## Headlines

1. **Phase 1 closes at the paper's scale: 38.1 % of 1,008 soups transition within 16k epochs (paper: 40 % of 1,000).**
   Median transition epoch 5,280; the histogram matches Figure 6's spread from 2k to 12k.
1. **Direction-flip brackets are dead: zero replicators in 20 million programs.**
   The brief's favoured primitive fails its first test; cyclic nearest-bracket beats the reference (18 vs 10 replicators per 10 million).
1. **Cyclic soups transition in 88 % of 384 seeds by 16k epochs**, against 38 % for the reference.
   The principled, total, halt-free variant is a gain, not a cost, on the emergence axis.
1. **Space slows emergence 25× and does not prevent it.**
   Three of four toruses transition, at 48k, 159k and 205k epochs; every takeover persists; a planted replicator takes over a torus in 3,000 epochs as a growing colony.
1. **Coexistence is a null result** seeded against seeded: the torus and the mixed soup keep comparable numbers of lineages.
1. **The grid machine exists** (`BFFGrid`, any dimension, both controls, 1D instance exact against the interpreter) **and it cannot yet sustain a planted replicator.**
   Five causes found and fixed in code through the night; the one that remains is that overlapping windows execute concurrently, where BFF gives every byte to one pair at a time.
   One design left to try: locked windows.

## Phase 1: the statistic at the paper's scale

`runs/phase1`, 1,008 seeds of 2^17 programs, 16,384 epochs, `bff_noheads`, mutation 1/4096, eight soups per process on six GPUs, about four hours.

| | paper | here |
| --- | --- | --- |
| transition within 16k epochs | 40 % of 1,000 | 38.1 % of 1,008 (95 % interval 35.1-41.1 %) |
| transition epoch, quartiles | spread 2k-12k | 2,480 / 5,280 / 10,896 |

Histogram: `notes/experiments/results/phase1_transitions.png`.
With the exactness checks (`01`, `05`), seeded takeover (`02`) and persistence (`06`), Phase 1 is done.

## Phase 2: which loop primitive

Programs tested per replicator found, 10 million programs per cell, detector score ≥ 60 (`03`):

| control | heads | uniform bytes | 50 % instruction bytes |
| --- | --- | --- | --- |
| matched (reference) | two | 0 | 10 |
| cyclic | two | 0 (one at score 32) | 18 |
| flip | two | 0 (best score 2) | 0 (best score 4) |
| any | swap | 0 | 0 |

Then cyclic soups at scale (`runs/phase2/cyclic_stat`, scored with the detector since entropy is blind for total machines): 244 of 384 replicator-dominated by 8k epochs and 337 of 384 (87.8 % ± 3.3) by 16k, against 38.1 % for the reference.
The replicators are ordinary BFF replicators, palindromic tapes like `[<,} ] },<[` with their mirror images; they score 64 under the reference semantics too.
High-order entropy is not a transition signal for total machines: smearing loops push it past 3 within 64 epochs.

Decision: `cyclic` is the principled variant to build on; `flip` and the swap-heads dialect are dropped.

## Space: tapes on a torus (`08`)

- `BFF(grid=(H, W))`: every program pairs with one of its four neighbours through a perfect matching, so interaction counts equal the control's.
- Same seed, same soup: random pairing transitions at 1,536 epochs; local pairing shows no drift at all for 32k epochs.
- To 262k epochs: three of four toruses transition (48k, 159k, 205k), one never; all persist.
- Seeded takeover on the torus: a colony with a ragged front covers 256 × 512 in 3,000 epochs and splinters into tens of thousands of species.
- Cyclic on the torus: no replicators in four soups by 32k epochs, where the mixed cyclic soup has four of four by 16k.
  Locality stalls the principled primitive too.
- Coexistence, seeded against seeded, four soups a side, effective species by instruction sequence at 16k epochs: 498 torus vs 321 mixed.
  Even.

## The grid machine (`09`)

`cax.cs.bff.grid.BFFGrid`: memory is one byte torus of any dimension; a thread is a position, a direction in `Z_{2d}`, two heads; brackets turn (`flip`, with `]` reflecting or quarter-turning) or jump along the thread's line (`cyclic`); heads move along the thread's axis; synchronous writes with random-priority conflicts; lifetime, respawn and mutation as BFF.
Nine tests; the 1D instance equals `run(control=...)` byte for byte for both controls.

What ran, all 512² unless noted, lifetimes of 8,192 steps:

| run | control | threads | start | lifetimes | outcome |
| --- | --- | --- | --- | --- | --- |
| g512, g2048 | flip, quarter turn | 1/64 | random | 1,200 / 850 | inert: memory stays random |
| r512, r2048 | flip, reflect | 1/64 | random | 9,000 / 5,000 | smears; instruction fraction 0.04 → 0.09; no repeats |
| c512, c1024 | cyclic | 1/64 | random | 3,000 / 500 | streaks along rows and columns; no replicators |
| ce512, re512 | cyclic, flip | 1/64 | 50 % instructions | 500 / 800 | instructions erased to 7-9 % within 4 lifetimes |
| cs512, cs1024 | cyclic | 1/4096, 1/1024 | 50 % instructions | 15,000 / 2,000 | slower decay, to 7-10 % instructions; no repeats |
| cw512 | cyclic, window 128 | 1/64 | random | 8,000 | instructions *rise* to 14 % and hold; no repeats |
| rw512 | flip, window 128 | 1/64 | random | 21,000 | instructions rise to 12 %; no repeats |
| pw512 | cyclic, window, planted replicator | 1/64 | random | 200 | the planted tape is destroyed, no copies |
| pa512, ca512 | cyclic, window, anchors (weave), planted / not | 1/64 | random | 128 | destroyed again; detector on 2,048 weave tapes: max score 0 |

The diagnosis: BFF's pair is a sandbox, a grid thread is not.
A smearing loop writes along its whole line for a lifetime; at one thread per 64 cells the memory is rewritten several times per lifetime, a per-cell mutation rate thousands of times BFF's.

The fix built at 03:00: `window`, a box of 128 cells around each thread's spawn point that its pointer and heads wrap within, and that the cyclic search walks.
Every thread gets a BFF-sized sandbox without tapes; sandboxes overlap where threads spawn near each other; the 1D instance with window 128 is still exact against the interpreter.
It changes the physics: instruction density climbs from 4 % to 14 % and holds, which is the drift a BFF soup shows before it transitions, instead of decaying to 7 %.
No replicator by morning in 30 million windowed thread-lives (cyclic) or 90 million (flip); the 1D rate under bytes this poor in instructions is below one in 10 million, so the budget is not yet conclusive.

The seeded test at 05:00 said more: a replicator from the cyclic soups planted along a row in a windowed grid is destroyed within a hundred lifetimes.
Two causes, both fixed in code but not yet in outcome: a thread starts anywhere heading anywhere, where BFF starts every execution at byte 0 of a program, hence `anchor`; and with anchors on a lattice each tape was executed 64 times per lifetime instead of once, hence anchors along the heading only, a weave of 4,096 row tapes and 4,096 column tapes per 512² that overlap by half.
The weave keeps the smearing and destroyed the planted replicator too, at 128 lifetimes, with the detector finding nothing among 2,048 of its tapes.
The remaining difference from the cyclic soup, where the same replicator dominates four of four runs, is that every cell is in a column tape as well as a row tape, so a tape is damaged twice per lifetime, and that the pair's random first half overlaps a neighbouring tape.
Three more causes were found and fixed in the following two hours (`09`): threads started anywhere, where BFF starts at byte 0 (anchors); a backward thread ran a different program from the mirror (relative heads); and the cyclic search counted the wrong bracket kind for backward threads (a bug, now tested).
With all of them fixed the planted replicator still left no copies on a ring of overlapping tapes.
The difference that remains is concurrency: in BFF every byte belongs to one executing pair at a time, and in any overlapping-window design a tape is copied into while another thread executes it.
Exclusive sandboxes that only shift between lifetimes are BFF pairs with local pairing on a lattice, which `BFF(grid=...)` already is.
This is where the grid stands: a tested, dimension-agnostic machine, a detector for it (`weave_score.py`), every departure from the exclusive pair tried and diagnosed, and one design left untried that keeps shared memory and exclusivity: windows that lock their cells for a lifetime.

## Also done

- Pallas kernel is the official GPU path (tag `bff-kernel-v1`), about 2× cubff on a random soup and 8× after takeover.
- `skeleton_hash` species, species statistics by skeleton and by instruction sequence, `phase1_stats.py`, `grid_inspect.py`, `grid_run.py`, `soup_run.py --grid/--pairing/--seed_program`.
- Example notebook `70_bff.ipynb` has the torus and the grid machine, executed; API docs list the new classes.
- Commits on `bff`: from `f3c2400` to the tip listed in `git log`.

## What I would do next

1. Locked windows: a thread's window is exclusive for its lifetime and spawns are rejected where a lock is held.
   Shared memory, any dimension, and BFF's one-pair-per-byte invariant; a morning's work, judged by `weave_score.py` on the planted ring first.
1. If that sustains a planted replicator, emergence in the weave from random memory at the cyclic soup's rate.
1. The cyclic soup's 1,000-seed statistic: launched at 05:00 on four GPUs, batches land through the morning; score with `replication_score` since entropy is blind for total machines.
1. A 2D replication detector, so the grid is judged by the same instrument as the soup.
