# GPU throughput: CAX vs cubff on H100, and whether to shard

Date: 2026-09-16, first day on the 8 × H100 box.
Everything below was measured here; the laptop numbers it supersedes are in `02_phase1_results.md`.

## Setup

- `jax[cuda12]==0.11.1` in the worktree venv; the host has no nvcc, so cubff (commit f212e84) was built for `sm_90` inside `nvidia/cuda:13.0.1-devel-ubuntu22.04` and runs on the host (cudart is static; brotli is the host's).
- The four CPU oracle harnesses build on the host with `g++ ... common.cc -lbrotlienc -lbrotlicommon`.
  `fidelity.py`, `detector_fidelity.py` and `replay.py` now read the binaries' location from `CUBFF_BIN` / `CUBFF_REPLAY` instead of a hard-coded laptop path.

## Exactness on the GPU build

All three checks from `01_reference_semantics.md` were rerun against the freshly built cubff, with the JAX side on an H100.

| Check | Size | Result |
| --- | --- | --- |
| `Bff::Evaluate`, uniform and enriched tapes, both head conventions, 8192 steps | 16,384 tapes | 0 mismatches (tapes and op counts) |
| Whole epochs with cubff's RNG replayed, both conventions | 1024 programs, 4 epochs | 0 mismatching bytes |
| `CheckSelfRep` with cubff's noise, both conventions | 204 programs | 0 mismatches |

The interpreter is exact on GPU as it was on CPU.

## Epoch time on one H100, random soup, `bff_noheads`

Seconds per epoch, steady state (first block excluded), one process per GPU.

| Programs | CAX | cubff (CUDA, same GPU) | ratio |
| --- | --- | --- | --- |
| 8,192 | 0.32 | 0.0068 | 47× |
| 131,072 (the paper's soup) | 0.42 | 0.012 | 35× |
| 524,288 | 0.64 | 0.034 | 19× |

Both agree on the physics: at epoch 128 of the 2^17 soup, high-order entropy is 0.169 in CAX and 0.168 in cubff (different RNG streams).

For scale: 16k epochs of the paper's soup pre-transition is about 2 h per seed in CAX and 3 min in cubff.
The CAX figure worsens after a transition, when most tapes survive compaction.

## Where the CAX time goes

`run` is a `fori_loop` over 8192 steps; the epoch cost is a per-step cost times 8192, and the per-step cost has a floor that does not depend on the batch.

| Pairs per step | µs per step (matched, 256 steps, no compaction) |
| --- | --- |
| 1,024 | 42 |
| 8,192 | 56 |
| 65,536 | 96 |
| 262,144 | 217 |

So the marginal cost of a pair-step is about 0.7 ns and the fixed cost of a step is about 40 µs.
After compaction (8% of a 2^17 soup, i.e. ~5k pairs) the remaining 7936 steps run at the floor: 7936 × ~45 µs ≈ 0.36 s of the 0.42 s epoch.

Decomposition of the floor at 8192 pairs (256 steps, each variant exact for what it computes):

| Variant | µs per step |
| --- | --- |
| `matched`, as shipped (cond between buffered and full search) | 53 |
| `flip` (no bracket search at all) | 14 |
| `matched`, always the full search | 54 |
| `matched`, always the buffered search | 58 |

The `lax.cond` costs nothing and the gather buffer buys nothing on GPU; the bracket search itself is 40 of the 53 µs, at any batch size up to 65k.
It is not the prefix sum: `cumsum`, `associative_scan` and a triangular matmul all give the same step time in the loop.
Nor is it launch latency on the host: `fori_loop` unrolling (4 to 64) and XLA command buffers (CUDA graphs, with `WHILE` and `CONDITIONAL` captured) change nothing.
The compiled loop body has about 40 kernels for `matched` and 20 for `flip` (fusions, gathers, scatters, reductions), each of about 1 µs of GPU time, and XLA cannot fuse across a gather or a scatter.
No setting fixes that; only a different program structure does.

## The kernel

`kernel.run_kernel` is one Pallas (Triton) kernel that does what cubff does: a block of 32 tapes (one warp) loops over the whole step budget on-chip and exits as soon as its last tape halts.
Per step it gathers the byte under the instruction pointer and both heads, decodes through the opcode table, writes at most one byte, moves the heads and the pointer, and, only when some tape in the block takes a jump, walks outward from the bracket exactly as the reference's loop does, stopping when every searching tape has its answer.
No compaction, no buffers, no `cond`: the halting that the XLA scan had to fake with masks is real.
`run(..., implementation="auto")` picks it on GPU and the XLA scan elsewhere; `interpret=True` runs it on CPU for the tests.

Exactness: 24 combinations (ASCII and swap-heads tables × three control flows × both head conventions × uniform and enriched tapes, 2048 tapes each, 8192 steps for `matched`) against the XLA scan, 0 mismatches in tapes, step counts and op counts; and the XLA scan is exact against cubff.

| Pairs, 8192 steps | XLA scan | kernel (block 32) | block 64 | block 128 |
| --- | --- | --- | --- | --- |
| 8,192 | 379 ms | 8.5 ms | 15.8 ms | 25.7 ms |
| 65,536 | 449 ms | 19.3 ms | 51.3 ms | 80.7 ms |

Smaller blocks win because a block lives as long as its longest tape and 6% of random tapes use the whole budget; one warp per block is the minimum.

Epoch of the paper's 2^17 soup on one H100, brotli included: **21 ms** (was 420 ms; cubff 12 ms).
16k epochs pre-transition is now about 6 minutes per seed, so the paper's 1000-run statistic is a day of one GPU or an afternoon of eight.

## Using 8 GPUs

Three ways to use more than one GPU were measured or reasoned through, all before the kernel; the conclusions survive it because the kernel makes the per-device work smaller, not larger.

1. **Several processes on one GPU.**
   Four full-size runs sharing GPU 6 each took 1.97 s per epoch (0.49 s per run-epoch): no gain over one process, and the GPU reports 100% utilisation.
1. **Several soups vmapped in one process.**
   Two 2^17 soups under `nnx.vmap` (with `nnx.split_rngs`) took 3.4 s per epoch, 1.7 s per run-epoch, four times worse than one soup.
   Under `vmap`, `lax.cond` becomes a `select` that evaluates both branches.
   Rejected; `jax.shard_map` over a run axis would be the tool if many soups in one process were ever wanted.
1. **Sharding one soup across devices**, implemented as `BFF(shard_axis="program")`.
   The mesh comes from `jax.set_mesh(jax.make_mesh((n,), ("program",), axis_types=(AxisType.Auto,)))`; `init_state` places the soup with `device_put(P("program"))`; each epoch gathers the pairs and mutates them under the partitioner, runs them device-locally under `jax.shard_map` (a Pallas call cannot be partitioned automatically), and writes back through the inverse permutation as a gather.
   One epoch is bit-identical to the unsharded epoch (checked at 2^17 and in `test_bff_sharded_epoch_matches_unsharded`).
   The first version wrote back with a scatter and cost about 800 ms per epoch at every soup size; the partitioner handles a sharded scatter badly.
   Timing with the gather, milliseconds per epoch, random soup:

   | Programs | one H100 | 8 H100s, sharded | speed-up |
   | --- | --- | --- | --- |
   | 2^17 | 20.6 | 12.0 | 1.7× |
   | 2^20 | 106 | 24 | 4.4× |
   | 2^22 | 416 | 62 | 6.8× |

   As predicted, it pays for large soups and not for the paper's; the fixed cost is the two all-gathers of the soup per epoch.

Conclusion: for the statistics we want (fraction of runs that transition, takeover persistence across seeds) the right use of 8 GPUs is 8 independent processes, one per device, as `04_server_plan.md` says; for one very large soup, `shard_axis` is ready and scales.
`pmap` is not used anywhere.

## Next

- Phase 1 at full scale: 8 seeds × 16k epochs, one per GPU, six minutes each pre-transition.
