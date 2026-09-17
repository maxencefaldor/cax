# The grid machine: one substrate, dimension as a parameter

Date: 2026-09-17, night.
Status: design decided and implemented as `cax.cs.bff.grid.BFFGrid`; everything here is open to revision by the first results (`10_grid_results.md`).

## What it is

Memory is one byte array of shape `(*dims)` on a torus: a line, an image, a volume.
There are no tapes and no pairing.
Threads walk the memory: a thread is a position (the instruction pointer), a direction, and two head positions.
Every step, every thread reads the byte under its pointer, executes it, and moves one cell in its direction.
Threads interact only because they share the memory: one thread's write is the next thread's code.
The state is literally the picture.

## Semantics (the `flip` machine of Phase 2, unfolded)

The instruction set is BFF's, unchanged, ten instructions, one byte each; the table is the parameter it always was.
What changes is the meaning of "direction" and of the brackets:

- **Direction** is an index in the ring `Z_{2d}`: directions `0..d-1` are `+axis`, `d..2d-1` are `-axis`.
  A turn is `±1` in that ring.
  In 1D the ring has two elements and any turn is a reversal: the `flip` machine exactly.
  In 2D a turn is a quarter turn, `+x → +y → -x → -y`; in 3D the six directions form one cycle.
  In every dimension `2d` turns of the same sense close a path, so a loop is a closed rectilinear path.
- **Brackets turn.**
  `[` turns left when the byte under head0 is zero; `]` turns right when it is nonzero; otherwise they are no-ops.
  Nothing is searched, nothing halts, nothing nests.
  A `while` loop in 2D is a room with a `]` at each corner: the pointer circulates while the test byte is nonzero and leaves straight through the first corner whose test reads zero.
- **Heads move along the thread's axis**, with absolute sign: `>` moves head0 by `+1` along whichever axis the thread is currently travelling, `<` by `-1`; `}` and `{` the same for head1.
  In 1D this is exactly BFF's head movement.
  Head moves relative to the *signed* direction were rejected: in a bounce loop the backward pass undoes the forward pass's moves exactly, so no bounce loop could ever copy anything.
- **Writes** are BFF's: `+`/`-` at head0, `.` copies head0 to head1, `,` copies head1 to head0.
  At most one byte per thread per step.
- **Totality**: every instruction is total; the only end of a thread's life is its step budget.

## Threads

- A fixed population of `num_threads` threads, default one per 64 cells.
- A thread lives `num_steps` steps (8192, BFF's budget) and then respawns at a uniformly random cell with a random direction and both heads on its pointer, as BFF starts a tape with everything at zero.
  Lifetimes are staggered at initialisation so respawns are spread over time.
- A respawn is the grid's version of "test a random program": the thread starts executing whatever it lands on.
  With `N` threads and budget `T`, the machine tests `N` random programs per `T` steps; BFF tests 65,536 per epoch of the paper's soup, so `N` should be of that order for a comparable appearance rate, which at one thread per 64 cells is a 2048 × 2048 image.
  Step cost is flat in `N` up to about that size, so this is free.

## Write conflicts

Synchronous update.
Every thread reads the memory as it was at the start of the step; all writes land together.
When two threads write the same cell, the winner is decided by a random per-thread priority drawn each step: the write is encoded as `priority << 8 | value` and scattered with `max`.
Deterministic, order-independent, and no thread has a standing advantage.

## Mutation

As in BFF: every `num_steps` steps each cell is replaced by a random byte with probability `mutation_rate` (1/4096).
Per-step mutation at rate `1/4096/8192` was rejected as too small to sample from a float32 uniform.

## Rendering

The memory is the image: the ten instructions in ten hues, data bytes in grey by value, the zero byte near black.
Threads are drawn as white dots at their pointers.
In 1D the image is the tape as one row.

## What the 1D instance is

`BFFGrid` with a 1-dimensional memory of length 128, one thread starting at cell 0 with direction `+x` and heads at 0, run for `num_steps` with no mutation, computes the same function as `run(tape, control="flip")` on that tape.
`test_grid_1d_matches_flip` checks it byte for byte.
This is the sense in which the principled BFF is "recovered exactly".

## Open questions this design does not settle

- Whether 2D-native replicators exist at a findable rate.
  A 1D program does not transfer: its brackets turn in 2D.
  The appearance-rate sweep for `flip` in 1D is a lower bound on the difficulty; a 2D detector does not exist yet.
- Whether the turn should be a quarter turn or a reversal.
  Reversal keeps every thread on its line forever, so 2D would be a bundle of independent 1D machines; rejected for that reason, not by measurement.
- Thread density, lifetime and mutation are BFF's numbers for comparability; none has been varied.
- Performance: this is an XLA implementation, one scatter and three gathers per step, roughly 50-100 µs per step regardless of thread count.
  A Pallas kernel would need one launch per step for the grid-wide barrier; not built.

## Revisions the same night

- **Quarter-turn brackets are inert in 2D** (`runs/grid3/g512`, `g2048`): after a thousand lifetimes the memory is still random, because a loop needs `2d` corners and random memory has none.
  `turn="reflect"` (the default now) makes `]` a reversal in every dimension, so the two-bracket bounce loop of the 1D machine exists on any line, and `[` alone changes axis.
  The 1D reduction is unchanged.
- **Flip finds no replicators even in 1D** (`03_phase2_design.md`: zero in 2 × 10^7 programs, against 18 for cyclic).
  The grid therefore has `control="cyclic"`: brackets jump along the thread's line with nesting and wrapping, threads never turn, and 2D is a weave of row and column machines sharing every cell.
  Its 1D instance is the cyclic machine exactly.
  Cost: a line of memory per thread per step; 1.4 s per lifetime on 512² with 4,096 threads, 8 s on 1024² with 16,384.
- **The shared memory is rewritten faster than anything can persist** (`runs/grid3/ce512`, `re512`, 02:50 London).
  Starting from memory with 50 % instruction bytes, both controls fall to 7-9 % instructions within four lifetimes and stay there; the equilibrium does not depend on the start.
  The cause is the loss of BFF's sandbox: a pair's execution can only damage its own 128 bytes, but a grid thread's smearing loop writes along its whole line for 8,192 steps, and at one thread per 64 cells the population rewrites the memory several times per lifetime.
  That is a per-cell mutation rate thousands of times BFF's, and no replicator can persist in it.
  Levers, none yet measured: thread density (runs `cs1024`, `cs512` at one thread per 1,024 and per 4,096 cells), lifetime, a write budget per thread, or heads confined to a window around the thread, which would give back a sandbox without tapes.
- **Where a thread starts matters as much as where it may write** (`runs/grid3/pw512`, 05:00 London).
  A replicator from the cyclic soups planted along a row in a windowed grid left no copies after 128 lifetimes.
  In BFF every execution starts at byte 0 of a program with the heads there too; a grid thread respawns at a random cell heading a random way, so it starts a planted program at its first byte once in ~256 spawns while other threads' windows overlap it every lifetime.
  `anchor` (05:10 London): threads spawn only on a lattice, and with a window the sandbox is the stretch of memory in front of the thread, so a thread starting on an anchor heading along a row executes the 128 cells ahead exactly as a BFF pair, and the pairs overlap by half along rows and columns.
  Runs `pa512` (planted) and `ca512` (unplanted).
- **Anchors are along the heading only** (05:30 London): a thread spawning to travel along axis `k` starts at a multiple of `anchor` on that axis and anywhere on the others.
  With `anchor=64, window=128` on 512² that makes 4,096 row tapes and 4,096 column tapes of 64 bytes, every cell in one of each, executed as BFF pairs that overlap by half; 4,096 threads give each tape about one execution per lifetime as a first half, as BFF gives each program one pair per epoch.
  The first anchored version used a 64 × 64 lattice, which is only 64 tapes in 512², each executed 64 times per lifetime; under `cyclic` a random first half runs the full budget and wrecks its second half, so the planted replicator was destroyed faster than it copied.
- **The cyclic replicators copy their mirror image.**
  The dominant tape of the cyclic soups writes its reverse into the second half of the pair, and the reverse writes the original, which is why the soups' dominant tapes come in mirrored pairs; any count of copies must look both ways.
- **Backward execution is not mirror execution** (06:30 London).
  Rows only, one thread per anchor slot as in BFF, the planted replicator still died (`runs/grid3/p1d`, first version).
  The cyclic replicators produce their mirror image, and the mirror is a replicator when executed forward; but a thread heading the other way executes the original *backwards* with heads that still move with absolute sign, which is a different program.
  So with both headings every tape got half a copying execution and one and a half damaging ones per lifetime, against BFF's half and half.
  `headings="positive"`: threads spawn heading `+axis` only; `flip` brackets may still reverse them.
  Planted ring and weave relaunched with it.
- A windowed bracket search now walks the window, not the whole line: 0.5 s per lifetime on the 262,144-cell ring instead of 200 s.
- **A bug in the cyclic search for backward threads, and relative heads** (07:00 London).
  The search chose which bracket kind to count from the walk direction instead of from the bracket under the thread, which is the same thing for a forward thread and wrong for a backward one; every run with both headings under `cyclic` had backward threads jumping to the wrong partner.
  Fixed, with a test that a backward thread over a mirrored tape produces the mirrored memory.
  That test needs `heads="relative"`: head moves follow the thread's heading, so running a stretch backwards is exactly running its mirror image forwards, and a replicator on a line is copied by threads heading either way.
  On the ring a chain of copies then grows at both ends and refreshes its interior from both sides, where with absolute heads it grew at the front and eroded from the back (`p1d`, earlier versions: single-thread checks showed the copy chain mirror, original, mirror working and a random predecessor changing 16 bytes of the tape per execution).
  Planted ring and weave relaunched with relative heads and both headings.
- **The remaining difference is concurrency, and it is fatal** (07:15 London).
  With relative heads and both headings the planted replicator still left no copies on the ring (`p1d`, 4,096 tapes scored, max 0).
  What BFF has that no overlapping-window design has: at any moment every byte belongs to exactly one executing pair.
  On the ring a cell is inside two windows, in the weave four, executed concurrently: a tape is copied into while another thread is executing it, and a half-written copy is what the next thread runs.
  Exclusive sandboxes with overlap only across lifetimes are BFF pairs on a lattice with local pairing along every axis, which is what `BFF(grid=...)` already is.
  So the night ends with the grid machine as a tested, dimension-agnostic module whose every departure from BFF's pair, shared memory, free spawning, overlapping windows, has been tried and diagnosed, and with the evidence pointing back to the exclusive pair as the unit that evolution needs.
  The one design not yet tried that keeps a shared memory and exclusivity: windows that lock their cells for the thread's lifetime, with spawns rejected where a lock is held.
