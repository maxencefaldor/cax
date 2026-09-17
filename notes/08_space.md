# Space, first step: BFF tapes on a torus with local pairing

Date: 2026-09-16 night to 2026-09-17 morning.
Code: `BFF(grid=(height, width))`, `local_pairing`, `skeleton_hash`, the grid `render`; runs with `soup_run.py --grid H W --pairing local`.
Pictures and videos: the gallery page (link in `10_night_report.md`).

## Why this first

The roaming-thread grid (`09_grid_design.md`) differs from paired BFF in two ways at once: space, and shared memory without tape boundaries.
Adding space alone to the paired soup isolates the first, costs nothing (the kernel runs the pairs exactly as before), and is the reference's own 2D setting.

## What was built

- The soup is a `height × width` torus of 64-byte programs in row-major order.
- Each epoch pairs every program with one of its four neighbours: a random axis and a random parity give a perfect matching, so every program interacts exactly once per epoch, as under random pairing; the order within a pair is random.
  The reference's 2D pattern (`make_2d_pattern.py`, radius 2) is a greedy matching that leaves some programs idle each epoch; the perfect matching keeps interaction counts identical to the control.
- `skeleton_hash` names a program's species: the hash of its instruction skeleton, so data-byte mutants of a replicator share a species.
  The log gains `species` (distinct skeletons) and `largest` (share of the largest).
- The grid render: one pixel per program, hue from the species hash, brightness from instruction density.

## Result 1: locality stalls emergence

Same seed, same soup at epoch 0, one difference: who each program is paired with.

| pairing | epoch | byte entropy | high-order entropy | species |
| --- | --- | --- | --- | --- |
| random | 256 | 7.995 | 0.26 | 88,203 |
| local | 256 | 7.998 | 0.02 | 94,965 |
| random | 1024 | 7.94 | 0.36 | 99,086 |
| local | 1024 | 7.99 | 0.03 | 97,513 |
| random | 2048 | 5.55 | 3.96 (transitioned at 1536) | 1,200 |
| local | 2048 | 7.99 | 0.03 | 99,073 |

Four local-pairing seeds run to 32,768 epochs: none drifted (high-order entropy 0.02 throughout, byte entropy 7.97).
The four random-pairing controls: two transitioned (epochs 1,536 and 11,584), two drifted the usual way (byte entropy 7.50 by epoch 8k).

The pre-transition drift is proto-replication.
Under random pairing a partial copier seeds copies of itself into new partners every epoch and grows exponentially; under local pairing it can only grow along the perimeter of its colony, and its neighbours are soon its own copies.
This is the 2026 paper's "pairing is what lets replicators take over", seen from the other side: mixing is what makes the appearance hazard compound.

## Result 2: locality delays emergence, it does not prevent it

Four local-pairing soups to 262,144 epochs (`runs/grid2/local_long`, seeds 10-13, 55 minutes on one GPU):

| soup | first epoch with high-order entropy ≥ 1 | state at 262k epochs |
| --- | --- | --- |
| 0 | 47,872 | entropy 6.6, 15,689 species, largest 1.6 % |
| 1 | 205,056 | entropy 7.0, 4,686 species, largest 2.9 % |
| 2 | none | flat, entropy 0.02 |
| 3 | 158,976 | entropy 7.0, 4,602 species, largest 0.9 % |

Three of four transition, at a median near 160k epochs against 6k under random pairing (`06`, and the 1008-seed statistic): about 25 times slower, consistent with polynomial rather than exponential growth of the proto-replicator population.
Every takeover persisted to the end of the run.

## Result 3: once a replicator exists, space is no obstacle, and it keeps many lineages alive

The hand-written replicator planted at the centre of a random torus under local pairing (`runs/grid2/local_seeded`):

| epoch | high-order entropy | mean steps per pair | species | largest species |
| --- | --- | --- | --- | --- |
| 256 | 0.10 | 635 | 94,749 | 6.9 % (the empty skeleton) |
| 1024 | 1.35 | 1,287 | 92,576 | 4.5 % |
| 2048 | 4.90 | 3,936 | 61,807 | 0.9 % |
| 4096 | 6.12 | 5,526 | 42,270 | 0.5 % |

The colony is a growing disc with a ragged front; by epoch 3,000 it has wrapped the torus.
Inside it is a mosaic: no lineage holds more than 1 % of the torus at any time.

The same 4,096 epochs under random pairing, emergent rather than seeded (`runs/grid2/random_frames`, seed 0):

| epoch | high-order entropy | species | largest species |
| --- | --- | --- | --- |
| 2048 | 3.96 | 1,200 | 35 % |
| 4096 | 4.55 | 12,161 | 13 % |

The picture has no geography: the replicator appears everywhere at once, salt and pepper.

## Result 4: coexistence at equal epochs, seeded on both sides, is a null result so far

Four seeded soups each, local versus random pairing, 16,384 epochs (`runs/coexist`, seeds 30-33).
The seed took in one soup of four on each side (the paper's seeded rate is 22%); one random soup transitioned on its own at epoch 2,048.
Species by *instruction sequence* (same instructions in the same order, wherever they sit; `species_stats.py`), which merges shifted and data-mutated copies:

| soup | epoch | effective species | species ≥ 0.1 % | largest |
| --- | --- | --- | --- | --- |
| local, seeded (soup 0) | 4096 | 2,592 | 147 | 2.0 % |
| random, seeded (soup 0) | 4096 | 1,879 | 183 | 3.3 % |
| local, seeded | 16384 | 498 | 139 | 9.0 % |
| random, seeded | 16384 | 321 | 186 | 3.4 % |
| random, emergent (soup 2) | 16384 | 14 | 64 | 41 % |

Seeded against seeded, the torus and the mixed soup keep comparable numbers of lineages and both lose diversity at a similar pace.
The 100-fold gap seen earlier between the seeded torus and an *emergent* mixed soup was the seed, not the space: the hand-written replicator spawns a far more diverse population than an emergent one does.
One soup per side; this is a first look, and the coexistence metric itself (effective species by sequence) is provisional.

## Result 5: cyclic brackets on the torus do not transition either

Four soups, `control="cyclic"`, local pairing, 32,768 epochs (`runs/phase2/cyclic_local`, seeds 50-53).
Under random pairing the same machine is replicator-dominated in four of four soups by 16k epochs (`03`); on the torus the detector finds no replicators in any soup at 8k, 16k or 32k epochs (0 of 1,024 sampled tapes each).
High-order entropy sits at 2.4 throughout, which is the smearing floor of a total machine, not a transition.
Locality stalls the principled primitive as it stalls the reference.

## What this means for the grid machine

- Space is not a free lever.
  It slows emergence by an order of magnitude here, and the roaming-thread grid will inherit that unless its threads mix (long lifetimes, random respawn).
- Whether space does something for what comes after is not shown: seeded against seeded, the torus and the mixed soup keep comparable numbers of lineages.
  The mosaic in the pictures is real, but the mixed soup has as many lineages, only without geography.
  The coexistence question needs more soups per side, longer runs, and a settled lineage metric before the roaming-thread grid is judged on it.
- The seeded torus is the visual: emergence takes tens of thousands of epochs, a planted replicator takes over in three thousand and makes a mosaic.
