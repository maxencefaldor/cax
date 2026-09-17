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
