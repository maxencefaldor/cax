# Kernel hill-climb

Started 2026-09-16, evening.
Goal: make `kernel.run_kernel` faster in every regime without changing its output.
Harness: `experiments/kernel_attempts.py` (four regimes, exactness gate against the XLA scan across every control flow, head convention and opcode table, one CSV row per attempt in `experiments/results/kernel_attempts.csv`).
Reference on the same box and soups: cubff does the random regime in 10.3 ms per epoch and the transitioned one in 35 ms.

## Attempts

Milliseconds per 65,536 pairs × 8192 steps on one H100; every row passed the exactness gate unless marked.
The `cyclic` and `flip` columns (enriched regime, every lane runs the whole budget) exist from attempt 13 on.
Regenerated from `experiments/results/kernel_attempts.csv`; `experiments/plot_attempts.py` draws it.

| # | When (UTC) | Change | random | enriched | mid | final | mean | cyclic | flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20:43 | baseline: walk search, block 32, select-store | 18.38 | 29.66 | 20.80 | 31.29 | 25.03 |  |  |
| 1 | 20:52 | masked store (no read of the overwritten byte); search walks 8 positions per reduction | 13.99 | 24.77 | 19.38 | 23.74 | 20.47 |  |  |
| 2 | 20:58 | whole-array refs addressed by lane row instead of BlockSpec tiles; no padding | 14.01 | 24.62 | 19.22 | 23.15 | 20.25 |  |  |
| 3 | 21:00 | step loop restructured: per-lane finish, loop test once per unroll (unroll=1) | 14.06 | 24.78 | 19.29 | 23.19 | 20.33 |  |  |
| 4 | 21:02 | initial pc/heads computed in-kernel; only table and tapes as inputs [GPU shared with another job: invalid] | 30.51 | 25.01 | 19.54 | 25.78 | 25.21 |  |  |
| 5 | 21:03 | same, unroll=8 steps per loop test | 14.43 | 25.41 | 20.42 | 27.65 | 21.98 |  |  |
| 6 | 21:03 | initial pc/heads computed in-kernel; only table and tapes as inputs | 14.23 | 24.88 | 19.74 | 24.34 | 20.80 |  |  |
| 7 | 21:05 | row index carried through the loop instead of hoisted (Triton LICM cost) | 11.03 | 13.62 | 11.73 | 22.75 | 14.78 |  |  |
| 8 | 21:07 | dummy row for dead lanes (interpret-safe masked stores); row carried | 10.76 | 13.74 | 11.62 | 21.57 | 14.42 |  |  |
| 9 | 21:07 | search_chunk=4 (dummy row, row carried) | 9.48 | 13.52 | 9.86 | 20.49 | 13.34 |  |  |
| 10 | 21:07 | search_chunk=16 (dummy row, row carried) | 15.25 | 18.41 | 17.12 | 28.20 | 19.74 |  |  |
| 11 | 21:09 | search_chunk=2 | 8.70 | 13.94 | 9.37 | 21.50 | 13.38 |  |  |
| 12 | 21:09 | search_chunk=6 | 9.35 | 13.04 | 9.80 | 20.62 | 13.20 |  |  |
| 13 | 21:10 | search compares raw bytes against the two bracket bytes instead of a table gather (chunk 4) | 8.25 | 11.02 | 8.77 | 18.51 | 11.64 |  |  |
| 14 | 21:10 | compare search, search_chunk=8 | 9.94 | 12.54 | 10.83 | 18.14 | 12.86 |  |  |
| 15 | 21:10 | compare search, unroll=2 | 8.18 | 10.99 | 8.58 | 16.95 | 11.17 |  |  |
| 16 | 21:11 | compare search, unroll=4 | 8.21 | 11.19 | 8.65 | 18.87 | 11.73 |  |  |
| 17 | 21:21 | current default (chunk 4, unroll 2, compare search) with cyclic/flip timing | 8.18 | 11.01 | 8.74 | 18.65 | 11.64 | 67.05 | 3.05 |
| 18 | 21:21 | current default (chunk 4, unroll 2, compare search) with cyclic/flip timing | 8.18 | 11.01 | 8.74 | 18.65 | 11.64 | None | None |
| 19 | 21:23 | search_chunk=16, to read the cyclic column | 13.62 | 16.48 | 15.41 | 23.42 | 17.23 | 63.05 | 2.33 |
| 20 | 21:23 | search_chunk=16, to read the cyclic column | 13.62 | 16.48 | 15.41 | 23.42 | 17.23 | None | None |
| 21 | 21:26 | per-lane cache of the last forward/backward search, invalidated by writes in the walked range | 4.54 | 8.59 | 4.97 | 6.98 | 6.27 | 53.99 | 2.30 |
| 22 | 21:26 | per-lane cache of the last forward/backward search, invalidated by writes in the walked range | 4.54 | 8.59 | 4.97 | 6.98 | 6.27 | None | None |
| 23 | 21:27 | bracket census, cyclic only (skips ring walks with no partner byte) | 4.52 | 8.49 | 4.90 | 6.90 | 6.20 | 54.24 | 3.32 |
| 24 | 21:27 | bracket census, cyclic only (skips ring walks with no partner byte) | 4.52 | 8.49 | 4.90 | 6.90 | 6.20 | None | None |

## Findings

- **Vectorised search is 7× slower** (scratch `kernel_v2.py`): loading the warp's (32, 128) tile and matching through `jnp.cumsum` inside the kernel, only on steps where a lane jumps, gives 136 ms on random against 18 ms for the walk.
  The tile load and the in-kernel scan are far more expensive than the average walk, which is short because most brackets match nearby.
  So the walk is not the bottleneck it looked like; the cost model needs the ablation below.
- **Cost model from ablation** (scratch `ablation.py`, 65k pairs, 8192 steps):

  | Variant | random | enriched | final |
  | --- | --- | --- | --- |
  | `matched`, as shipped | 18.4 ms | 29.6 ms | 31.3 ms |
  | `flip`: no search, no halting, every lane all 8192 steps | 10.3 ms | 10.4 ms | 10.4 ms |
  | `cyclic`: search on a ring, no halting | 164 ms | 138 ms | 93 ms |
  | `matched` with the search stubbed to "no partner" (tapes die at the first jump) | 0.3 ms | 0.2 ms | 0.3 ms |

  So the step chain costs 10.3 ms for the full budget on every lane, in every regime: about 0.6 ns per warp-step, or 5 × 10^10 lane-steps per second, ten times cubff's rate per executed step.
  What we lose is elsewhere: the warp tail (random tapes halt at a mean of 625 steps but their warps run 8192) and the search (the difference between `matched` and `flip` on random is 8 ms of walking, mostly unmatched brackets that walk to the end of the tape and halt).
  The `cyclic` search is ten times worse still, because a ring has no end to bail out at and nothing halts; Phase 2 lives there, so the search matters twice.
- **Persistent lanes, static stride: no.** (scratch `kernel_persistent.py`) Lane g running rows g, g + L, g + 2L, ...
  for L total lanes is exact and needs no atomics, but it is slower the fewer lanes there are: at 2 tapes per lane random costs 12.9 ms against 9.4 ms at one tape per lane, and `final` 41 ms against 23 ms. A lane's total is a sum of heavy-tailed run lengths, so the slowest lane gets slower as lanes take more tapes; only a dynamic queue (claim the next tape when yours halts) balances that, and the atomic-counter version hung on the GPU (and interpret mode cannot run atomics).
  Parked, not dead: the tail is still 90% of the random-soup work.
- **Two incidental wins from the same prototype, 1.7× overall.**
  A masked store for the written byte instead of a load-select-store (no read of the byte being overwritten, no store at all for lanes that write nothing), and a search that visits 8 positions per warp reduction.
  At one tape per lane: random 9.4 ms (was 18.4), enriched 14.2 (29.7), mid 11.8 (20.8), final 23.3 (31.3).
  Ported as attempt 1.
- **Attempt 1 (ported): mean 20.5 ms** (random 14.0, enriched 24.8, mid 19.4, final 23.7), exact.
  Less than the prototype's 9.4 / 14.2 / 11.8 / 23.3 at one tape per lane; the remaining difference is how the tape is addressed (a whole-array ref with per-lane row indices in the prototype, a `BlockSpec` tile in the shipped kernel) and is being isolated.
- **Decode by compare chain: no.** 11 compares and selects on the byte cost more than the dependent table gather (random 14.4 ms against 9.4 in the prototype).
- **Vector atomics are silent no-ops here** (scratch `atomic_test2.py`): `plt.atomic_add` with a vector of indices, distinct or duplicated, leaves the counter untouched and returns zeros; a scalar atomic works.
  That is why the dynamic queue hung.
  A queue can still be built from scalar atomics: count the lanes that halted (warp reduction), claim that many rows with one scalar atomic, and hand them out by the lanes' prefix ranks.
- **Dynamic queue (scalar atomic + lane ranks): exact, and slower everywhere.**
  At 65k pairs the best grid (528 warps) gives random 28 ms, final 89 ms; the one-tape-per-lane static layout gives 11.4 and 23.0.
  The reason is occupancy, not the queue: 65k pairs are only 2048 warps, 15 per SM, so nothing is left to balance and the queue only adds an atomic and a prefix sum per step.
  Random-soup work is already at the `flip` floor (every warp runs the full budget: 11.4 ms against 10.3), so at this soup size the tail cannot be recovered by scheduling, only by making a warp-step cheaper.
  The queue would pay for soups with many more pairs than lanes (2^20 and up); parked with the code in scratch `kernel_persistent.py` (`QUEUE=dynamic`).
- **Whole-array addressing vs `BlockSpec` tiles:** at one tape per lane the prototype gives random 11.4 ms where the ported kernel (attempt 1) gives 14.0; the tile form costs 20%.
  Worth porting the whole-array form.
- **Attempts 2 to 5: same numbers as attempt 1** (whole-array addressing, the prototype's loop structure, 8-step unroll, initial state computed in-kernel).
  Unrolling even hurts `final` (27.7 ms).
  A body/wrapper swap between the shipped kernel and the prototype put the whole gap in the body, and a one-line bisection found it:
- **The row index must be a loop carry, not a loop invariant.**
  The shipped kernel computed `rows` once outside the step loop; the prototype carried `row` and recomputed the clamped row inside each step.
  Carrying it: random 11.2 ms (was 14.2), enriched 13.9 (was 24.9), exact.
  Triton hoists the invariant row addressing (base pointers for the gathers) out of the loop and the step pays for it anyway, most on jump-heavy soups where the search does many gathers.
  Ported as attempt 6.
- **Unrolling the step loop: no.** 8 steps per liveness test in the prototype: random 13.1 ms (was 11.4), final 28.9 (was 23.0); in the shipped kernel (attempt 4) the same.
  Register pressure from eight copies of the step body outweighs the saved reductions.
- **Interpret mode and masked stores.**
  With whole-array refs, dead lanes that alias a live row break the interpreter (it discharges a masked store as a read-modify-write scatter, and duplicate indices have no defined order) while the hardware is fine.
  Fixed by one dummy row at the end of the padded tape that every dead lane points at (attempt 7).
- **Search chunk: 4 beats 8 beats 16** (9.5 / 13.5 / 9.9 / 20.5 ms, mean 13.3, against 10.8 / 13.7 / 11.6 / 21.6 at 8 and 15.3 / 18.4 / 17.1 / 28.2 at 16).
  Most matches are within a few positions; a big chunk visits positions nobody needs.
  Default is now 4; 2 and 6 are being measured.
- **Masked load of the second head's byte only for `,`: no change** (within noise on all four regimes); not adopted.
- **Chunks 2, 4 and 6 are within noise** (means 13.4, 13.3, 13.2); 4 stays.
- **Bracket bytes by compare instead of table gather in the search: yes.**
  Two compares on the raw byte against the two bracket bytes (passed in as a 2-element input, computed from the table with `argmax`) replace a dependent gather per visited position: random 8.3 ms (was 9.5), enriched 11.1 (13.3), mid 8.7 (9.8), final 19.9 (23.2), exact.
  Valid because every opcode table maps each bracket from exactly one byte.
  Ported as attempt 9.
- **Attempt 9 (compare search in the shipped kernel): mean 11.6 ms**, random 8.3, enriched 11.0, mid 8.8, final 18.5.
  Ahead of cubff on the same box in every regime measured (10.3 random, 35 transitioned).
- **Unroll 2 now helps a little** (mean 11.2, final 17.0) where 8 hurt; with the cheaper search the body is small enough for two copies.
  Default is now 2; 4 is being measured.
  Chunk 8 with the compare search is worse than 4 (12.9), as before.
- **Unroll 4: worse than 2** (mean 11.7 against 11.2); 2 stays.
- **Full epoch of the paper's soup (`bench_kernel.py`, brotli excluded): 7.2 ms random, 19.7 ms transitioned**, of which the kernel is 6.3 and 19.1; the pairing gather, mutation and write-back are under a millisecond.
  This afternoon the epoch was 21 ms; cubff is 12.3 ms random and 35 ms transitioned on the same GPU.
- **Two-phase compaction (everything for 256–1024 steps, then only the survivors): exact, and slower** (scratch `kernel_twophase.py`, random 13.2 ms against 8.2).
  The 8% of survivors are 164 warps, one or two per SM, and a lone warp's step is about 1.7 µs of latency; 2048 warps only look fast because 16 per SM overlap.
  So the per-step *latency* is the target now, not the work: a step is a store followed by loads of the same tape line, and on Hopper a global store invalidates the L1 line, so the loads go to L2.
  Two consequences: bigger soups are nearly free per program (more warps per SM), and the next attempts prefetch the next step's inputs so a step's loads overlap the previous step's compute.
- **No-store ablation: inconclusive.**
  Removing the tape writes (wrong semantics, timing only) leaves random at 8.0 ms against 8.2, so store-induced L1 invalidation is not the dominant cost either; the per-step cost needs a profiler, not more ablations.
- **Prefetching the next instruction byte during the step: exact, slightly slower** (random 8.8 ms against 8.2).
  The extra load and the masked load cost more than the latency they hide, so the instruction-byte load is not the critical chain either.
- **Several soups per kernel launch: 2.4× throughput.**
  (scratch `batch_scaling.py`) One 2^17 soup is 2048 warps, 15 per SM, too few to hide a step's latency.
  Stacking the pairs of R soups into one launch (random regime): 1 soup 8.2 ms; 2 soups 5.6 ms per soup; 4 → 4.6; 8 → 3.8; 16 → 3.5.
  Exactness is untouched (a lane never sees another tape) and this is exactly what the many-seed experiments need, so `BFF.pair_and_run` now accepts a leading batch of soups and `soup_run.py --soups R` runs R seeds per process.
- **Packed-flags decode (one gather yields head deltas, write kind, flags): exact, slightly slower** (random 8.5 ms against 8.2).
  The compares it replaces were never the cost; ALU work is not the bottleneck at this occupancy.
- **Profiler:** Nsight Compute is not on the box and the apt repository only offers 2022 builds that predate the H100, so the climb stays empirical.
- **End to end with 8 soups per process** (`soup_run.py --soups 8`, brotli included): 4.9 ms per soup-epoch, so 16k epochs of 8 seeds take about 10 minutes on one GPU and 64 seeds fit in the same time on eight.
- **Two warps per block (block 64): much worse** (random 10.9 ms against 8.2 alone, 5.3 against 3.5 per soup at 16 soups; final 66 ms against 18).
  The block-wide reductions and the search now span 64 lanes and the tail doubles; one warp per block stays.
- **Phase 2 machines, timed from attempt 13 on** (enriched regime, every lane runs all 8192 steps): `flip` 3.05 ms, `cyclic` 67 ms, against 11.0 for `matched`.
  `flip` is the pure step chain, three times cheaper than the first kernel's 10.4.
  `cyclic` is the outlier: an unmatched bracket walks the whole ring every time and nothing ever halts, so the search dominates by far.
  Two attempts target it: a per-lane bracket census that skips the walk when no partner byte exists on the tape, and a larger search chunk for ring walks.
- **Bracket census (skip the walk when the tape holds no partner byte): exact; slightly slower on `matched` (random 8.7 ms against 8.2), a big win on `cyclic` final (18 ms against 40)** and nothing on `cyclic` random and enriched (75 against 82, 67 against 67).
  Evolved soups are full of tapes with `[` and no `]`, which on a ring walk the whole tape every loop; random tapes with partners still walk far.
  Kept in scratch (`kernel_census.py`) as a `cyclic`-only option; not in the default path.
- **Chunk 16 for the ring walk: 67 → 63 ms on `cyclic`, and 17 ms mean on the rest**; not worth a control-specific chunk.
  A power-of-two mask instead of the modulo in the ring walk: identical timings.
  The ring walks are long because partners are far or absent, not because a visited position is expensive.
- **Search cache: the largest win of the climb, exact.**
  Each lane keeps its last forward and its last backward search (bracket position, partner position) and reuses it when the same bracket jumps again, unless a write has landed in the range the walk visited (or on the bracket itself); a walk that found no partner visited everything, so any write invalidates it.
  A copy loop writes outside its own brackets, so it searches once and then never again.
  Scratch A/B against the shipped kernel: random 4.5 ms (was 8.2), enriched 8.6 (11.1), mid 4.9 (8.6), final 7.5 (20.1); `cyclic` random 40 (82), enriched 54 (67).
  Ported as attempt 15.
- **Cache and census together** (scratch `kernel_cc.py`): on `cyclic`, random 23 ms (cache alone 40, before both 82), final 8.1 (37), enriched 55 (unchanged, those tapes have partners far away); on the matched regimes the census costs about 0.5 ms (10%).
  So the census ships behind a `control == "cyclic"` guard (attempt 16) and the matched path keeps the cache only.
- **Attempt 15 (cache in the shipped kernel): mean 6.3 ms**, random 4.5, enriched 8.6, mid 5.0, final 7.0; `cyclic` 54, `flip` 2.3.
  Four times the baseline's mean; against cubff on the same GPU, 2.3× on random and 5× on the transitioned soup.
- **Direct check against cubff's binary after the cache and census** (`fidelity.py`, 16,384 tapes, both head conventions): 0 mismatches in tapes and op counts.
- **Two cache entries per direction: no.** Register pressure costs more than the extra hits: random 6.0 ms against 4.6, mid 6.4 against 4.9; only `cyclic` enriched gains (54 → 44).
  Not adopted, not even behind the `cyclic` guard.
- **Soups per launch, re-measured with the cache:** random 4.6 ms alone, 3.3 at 8 soups, 3.1 at 16; final 7.1 alone, 5.4 at 8, 5.1 at 16.
  End to end with 8 soups per process, brotli included: 4.5 ms per soup-epoch.
- The plot and the table are published as a page: <https://claude.ai/artifact/9x45eF1eD6q1RJNNkcU7RS>.

- **Chunk 16 with the cache: `cyclic` 54 → 45 ms, matched regimes +3%.** So the chunk is per control (4 matched, 16 cyclic), attempt 18.
- **Trimming control-constant arrays from the loop carry (direction outside `flip`, census outside `cyclic`): identical timings.** Triton already drops them; not adopted.

## Reading the kernel: where the time could go, and ideas

Per step, per warp, the chain is: load byte at `pc` → table lookup (dependent load) → decode → loads at `head0`, `head1`, and at the write position (dependent on the decode) → one store → head/pc arithmetic → warp reduction of `live` → loop back.
Three dependent L1 round-trips plus a reduction per step, with only ~16 warps per SM resident (2048 warps for 65k pairs over 132 SMs), so latency is barely hidden.

Ideas, roughly by expected payoff:

1. **Persistent lanes with a work queue (structural).**
   Today a warp lives as long as its longest tape, and 6% of random tapes use the whole budget, so 86% of warps run 8192 steps for a mean of 612 useful ones.
   Instead: each lane owns a tape; when it halts, the lane writes its result and fetches the next tape index from an atomic counter.
   Total work becomes the sum of steps, not 8192 × warps; on random soups that is ~10× less work, and the grid can be sized to fill the SMs.
   Exactness is untouched (each tape is still run alone, results land in its row).
   Needs whole-array refs instead of BlockSpec tiles and `pl.atomic_add`; both exist in Pallas Triton.
1. **Decode without the table load.**
   `cmd = table[byte]` is a dependent gather every step; with at most 11 distinct instruction bytes it can be 11 compares on the byte against scalars, removing one L1 round-trip from the critical path.
   Only valid when the table maps each op to at most one byte (true for every table we ship; verify in the wrapper and fall back otherwise).
1. **Fewer loads.**
   `value1` is only needed for `,`; `current` only when nothing is written.
   Loading the write position's byte can be replaced by a masked store (`plt.store(..., mask=live & writes)`) so the store needs no read.
1. **Check liveness less often.**
   The per-step warp reduction of `live` could run every 8 or 16 steps: an unrolled inner loop of steps with no reduction and no exit, then one test.
   Halted lanes just idle for a few steps; results are unchanged because halted lanes are masked anyway.
1. **Occupancy.**
   Only 2048 warps exist for 65k pairs; H100 wants ~64 warps per SM to hide latency, we have 16.
   Persistent lanes (idea 1) fix the tail but not this; only more pairs per epoch do, which is what sharding *away* from would suggest: bigger soups are nearly free per program.
1. **Search walk.**
   Cheap per iteration but each iteration is a dependent load plus a reduction; if the ablation says the search matters in the enriched/mid regimes, an unrolled fixed-length walk (e.g. 8 positions per iteration, one reduction) cuts the reductions 8×.

## Bigger picture

- The soup epoch has three non-kernel passes (gather, mutation, write-back gather) worth ~1 ms; not the priority.
- Post-transition soups are a different regime: every tape lives, loops constantly, and jumps every few steps; that is where Phase 2's total (`flip`, `cyclic`) machines live all the time.
  Any change must be measured there, hence the `final` and `enriched` regimes in the harness.
