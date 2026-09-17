# Beyond replication: instruments, room, and a resource

Date: 2026-09-17, day.
Status: the instruments are built and running on saved soups; genome room is running; the resource is a design, not code.

## Why

Every substrate so far plateaus after takeover, ours included.
Three reasons, each with a lever:

1. **The genome cannot grow.**
   A replicator lives in a 64-byte tape, so whatever it evolves into must fit in 64 bytes minus the copier.
   Lever: longer tapes.
1. **Copying is the only selected trait.**
   Once the soup is full, the fastest and most mutation-proof copier wins and selection stops.
   Lever: a resource, so that efficiency, theft and defence become traits.
1. **Mixing erases structure.**
   Lever: space, once there is something to specialise on; on its own it only slows emergence (`08`).

And a fourth reason we can fix at once: nobody has been measuring anything but copying.

## Instruments (built: `cax.cs.bff.assay`, `notes/experiments/soup_assays.py`)

Seven assays give every lineage a behaviour vector in [0, 1]:

| assay | the experiment | what a high value means |
| --- | --- | --- |
| `replicates` | detector, random partners | copies alone |
| `replicates_in_soup` | detector, partners drawn from the soup | copies on the soup's programs; high here and low alone is a parasite |
| `replicates_with_kin` | detector, partners are copies of itself | copies among its own kind |
| `survives` | second half of a pair, random first half | resists being run over |
| `survives_in_soup` | second half, first halves from the soup | resists the soup's programs |
| `overwrites_host` | first half, the dominant program as second half | damages or replaces the dominant lineage |
| `copies_partner` | first half, random partner | reproduces the other rather than itself |

Plus the mutational scan: per position, the fraction of single-byte mutants that still replicate.
Its zeros are the **core** (length and span), its mean the **robustness**.
`soup_assays.py` runs both on the top lineages by instruction sequence at every checkpoint, so each lineage has a trajectory, and the dominant lineage's hash per checkpoint gives the **turnover**.

Higher-order emergence, in these terms, is a lineage landing in a region of the seven-dimensional space that no lineage occupied before.
A parasite is high `replicates_in_soup`, low `replicates`.
A defended replicator is high `survives_in_soup`.
A predator is high `overwrites_host`, low `replicates`.

First runs: the cyclic pilot soups and a reference soup over 32k epochs (`runs/assays`).

## Genome room (running: `runs/long`)

256-byte tapes, 2^15 programs, cyclic and matched, four soups each, 65,536 epochs.
The question is one number: the core length of the dominant lineage over time.
If replicators grow to carry bytes that earn their keep, complexity is not capped by the tape and the premise of the programme holds.
If they stay near 60 bytes in a 256-byte tape, the cap was never the problem.

## A resource: steps as energy (design, not built)

The step budget is already a currency: a copy costs the steps it takes.
Make it explicit and scarce:

- Every program carries an energy count, part of the state next to its bytes.
- An epoch gives every program an income `E`.
- Executing a pair charges each step to the program whose half the pointer is in.
- A program whose energy is below zero at the end of the epoch is dead: its bytes are replaced by random ones.

What this buys, and nothing else does:

- **Efficiency is selected.**
  Two copiers now differ by how many steps they spend; the frugal one keeps energy for a bad epoch.
- **Theft pays.**
  A program that makes its partner's half do the copying spends nothing, which is the parasite of Tierra, and a program whose copying loop runs on in the partner's half bleeds the partner.
- **Death exists** other than being overwritten, so a lineage can lose without a rival replacing it, and empty room appears for new emergence.
- **Space becomes useful**, because parasites are what locality protects against.

Cost: the state becomes `(soup, energy)` rather than an array, and the kernel needs two step counters per pair, one per half.
That is a morning; the decision is whether `E` is a fixed income or proportional to the steps the program made its partner spend, which is a second-order choice to settle by measurement.

## Order

1. Read the first assay trajectories (today).
1. Read the core length at 65k epochs on 256-byte tapes (tonight).
1. Build energy if the assays show nothing beyond copying, which is the expectation.

## First readings (2026-09-17, evening)

`runs/assays/*.csv`, the most common exact tape of each soup at each checkpoint, assays under the soup's own control flow.

**Reference soup, 64-byte tapes, seed 0, epochs 4k to 32k** (`matched_seed0.csv`):

| epoch | dominant lineage | share | replicates | survives in soup | core length | robustness |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 146477 | 1.1 % | 1.00 | 0.66 | 8 | 0.86 |
| 8,192 | 146477 | 1.7 % | 1.00 | 0.77 | 8 | 0.84 |
| 12,288 | 549073 | 0.9 % | 1.00 | 0.44 | 9 | 0.84 |
| 16,384 | 424354 | 1.2 % | 1.00 | 0.22 | 9 | 0.82 |
| 20,480 | 620477 | 3.2 % | 1.00 | 0.11 | 9 | 0.82 |
| 24,576 | 541657 | 4.6 % | 1.00 | 0.04 | 9 | 0.83 |
| 28,672 | 832810 | 5.0 % | 1.00 | 0.05 | 10 | 0.81 |
| 32,768 | 217938 | 0.7 % | 1.00 | 0.10 | 10 | 0.81 |

- **Turnover is high**: the dominant lineage is replaced at almost every 4k-epoch checkpoint.
- **Every winner is the same kind of thing**: a copy loop with a core of 8 to 10 bytes and robustness 0.8, whatever its lineage.
- **Survival in the soup falls from 0.66 to 0.05**: over time every program is overwritten by whatever it is paired with, so the soup becomes a contest of overwriting, which favours the fastest copier and nothing else.
- No parasites (`replicates_in_soup` never exceeds `replicates`), no defence (`survives_in_soup` never rises), no cooperation (`copies_partner` stays at 0).

**Cyclic soups** (`cyclic_soups.csv`): the same picture with cores of 8 to 13 bytes; one soup converged to a single lineage at 45 % whose members overwrite each other (survival in soup 0.01).
The early "dominant" tapes of two soups are runs of a single byte written by smearing loops, inert, which is what the entropy signal was reading.

**Genome room, 256-byte tapes, reference control, 65k epochs** (`long_matched_seed64.csv`, two of four soups transitioned):

| epoch | soup 0 core | soup 3 core | robustness |
| --- | --- | --- | --- |
| 16,384 | 11 | 12 | 0.80 |
| 32,768 | 10 | 13 | 0.80 |
| 49,152 | 10 | 13 | 0.80 |
| 65,536 | 10 | 13 | 0.79 |

- **The room goes unused.**
  With four times the tape, the replicating core stays at 10 to 13 bytes for 65k epochs, and nothing else in the tape is conserved.
- So the cap on complexity was never the tape length; it is that nothing but copying is selected.

**Conclusion.**
With the current physics there is nothing beyond replication to find, in either machine, and the instruments would see it if there were.
That is the expected result, and it is the case for the resource: without something to be efficient at, to steal or to defend, selection has one axis.
Energy is next.
