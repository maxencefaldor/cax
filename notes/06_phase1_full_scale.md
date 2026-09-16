# Phase 1 at full scale: the paper's soup on H100s

Date: 2026-09-16, evening.
`soup_run.py`, `bff_noheads`, 2^17 programs, mutation 1/4096, 16,384 epochs, one seed per GPU, the Pallas kernel (`05_gpu_throughput.md`).
Runs stop 4,096 epochs after high-order entropy (hoe) first reaches 1, or at 16,384 epochs.

| Seed | Transition epoch (hoe ≥ 1) | hoe at end | mean steps per pair at end | epochs run | wall s |
| --- | --- | --- | --- | --- | --- |
| 0 | 2176 | 5.31 | 6222 | 6,272 | 154 |
| 1 | 6528 | 5.24 | 7374 | 10,624 | 247 |
| 2 | none in 16,384 | 0.12 | 616 | 16,384 | 292 |
| 3 | 2432 | 4.93 | 2861 | 6,528 | 125 |
| 4 | none in 16,384 | 0.12 | 624 | 16,384 | 290 |
| 5 | none in 16,384 | 0.12 | 661 | 16,384 | 291 |
| 6 | none in 16,384 | 0.13 | 627 | 16,384 | 287 |
| 7 | 12352 | 4.93 | 7301 | 16,384 | 331 |

## What it says

- **4 of 8 seeds transition within 16k epochs.**
  The paper reports 40% of 1,000 runs; 4/8 is consistent with that (95% interval 16–84%).
  Eight seeds is a pilot; the statistic needs ~100 seeds, which is now an afternoon of eight GPUs.
- **Every takeover persists.**
  All four transitioned soups sit at hoe 4.9–5.3 four thousand epochs later, with mean steps per pair 2,900–7,400 (random soups: ~600).
  The collapses seen at 8,192 programs (`02_phase1_results.md`) were a small-soup effect, as suspected; at 2^17 the paper's Figure 5 is reproduced.
- **Pre-transition epochs cost 18 ms, post-transition 28 ms** (seed 0: 2,176 epochs in 39 s, then 4,096 in 115 s).
  The kernel's warp-lives-as-long-as-its-longest-tape cost shows up as the 1.5× and no more.
- Transition epochs are spread from 2k to 12k, as in the paper's Figure 6.

## Next

- 100 seeds for the transition fraction, and time-to-transition in interactions to compare with the 8,192-program runs.
- Takeover persistence versus soup size (2^13, 2^15, 2^17) to pin the collapse.
