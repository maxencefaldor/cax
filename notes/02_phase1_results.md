# Phase 1 results: faithful BFF in CAX

Status: laptop-scale results, 2026-09-16.
The full-scale reproduction on H100s is in `06_phase1_full_scale.md` and the GPU throughput measurements in `05_gpu_throughput.md`; those supersede the throughput and emergence numbers here.
The exactness checks, the seeded-takeover rates and the collapse analysis remain the record.

## What was built

- `cax.cs.bff`: `BFF` complex system (state = soup array, step = epoch), `run` /
  `step` interpreter, `language` (byte-to-opcode tables: ASCII, permuted; parser),
  `detector` (port of `CheckSelfRep`), `metrics` (brotli high-order entropy).
- 20 unit tests, docs page, README row, example notebook `70_bff.ipynb`.
- Oracle harnesses and replay scripts in `notes/experiments/`.

## Exactness against cubff

| Check | Size | Result |
| --- | --- | --- |
| `Bff::Evaluate` on random and instruction-enriched tapes, both head conventions, 8192 steps | 16,384 tapes | 0 mismatches (tapes and op counts) |
| Whole epochs with cubff's RNG replayed (init, shuffle, mutation), both conventions | 1024 programs, 4 epochs | 0 mismatching bytes |
| `CheckSelfRep` with cubff's noise, both conventions | 204 programs | 0 mismatches |

## Throughput (Apple Silicon, XLA CPU, one process)

| Batch | Epoch, random soup, full budget |
| --- | --- |
| 4096 pairs (8192 programs) | 0.45 s |
| 16,384 pairs | 1.3 s |
| 65,536 pairs (2^17 programs, the paper's soup) | 3.7 s |

cubff single-threaded on the same machine: 0.13 s for 65,536 pairs.
Under a loaded machine the CAX numbers degrade several-fold (XLA's thread pool spins).

## Seeded takeover (paper Fig. 7 "seeded": one hand-written replicator, 128 epochs)

| Implementation | Programs | Runs | Takeover (hoe ≥ 1) | 95% CI |
| --- | --- | --- | --- | --- |
| paper | 2^17 | 1000 | 22% | |
| cubff (here) | 2^17 | 40 | 25% | 14–40% |
| cubff (here) | 8192 | 200 | 18% | 14–24% |
| CAX | 8192 | 6 so far | 0% | (pending, 60 planned) |

The outcome is bimodal in every run: hoe either stays below 0.3 or exceeds 3; the two thresholds in use (1 and 3) never disagree.

## Emergence from a random soup, N = 8192, mutation 1/4096 (cubff, 4 seeds, 400k epochs = 1.6e9 interactions each)

| Seed | Transitions (hoe ≥ 1) | Outcome |
| --- | --- | --- |
| 1 | epoch 129,921 (5.3e8 interactions) | takeover to hoe 5.9 by 170k, **collapse** to 0.1 by 180k, brief rebound at 184k, random again by 190k |
| 2 | none in 400k epochs | |
| 3 | epoch 231,105 (9.5e8) | hoe 4.2 for ~2k epochs, then **collapse**; nothing after |
| 4 | epoch 108,993 (4.5e8) and again ~337k | first one transient (peak hoe 1.8, gone within 2k epochs); second one at hoe 5.5 and running |

Every takeover at this soup size collapsed within ~10k epochs.
At 2^17 programs all four takeovers persisted (`06_phase1_full_scale.md`), so this is a small-soup effect: N = 8192 is not a faithful proxy for the headline statistic, and the collapse itself is the finding.
Checkpoints around the seed-1 collapse (172k, 176k, 180k, 184k) are being scored with the detector to tell whether the replicators die or drift out of the 64-byte frame.

## CAX continues cubff's takeover (cross-implementation dynamics check)

CAX resumed from cubff's seed-1 checkpoint at epoch 131,072 (hoe 2.85) with its own pairing and mutation RNG, 2,500 epochs.
At epoch 133,568 CAX reads 3.30 bits against cubff's own 3.44 at 133,505; over the whole window the two curves interleave within the noise of the takeover.
Log: `experiments/results/cax_resume_from_cubff_s1_131072.csv`.
Cost: 1.4 s per epoch on an evolved soup under a loaded machine.

## CAX emergence runs

Two seeds were run at N = 8192 concurrently with the cubff jobs and reached only ~700 epochs in 40 minutes because of CPU contention; stopped, to be rerun on an idle machine (expected 0.45 s/epoch, so ~16 h to 130k epochs).

## What the evolved replicators look like (cubff seed 1, N = 8192)

- Epoch 131,072 (hoe 2.85, mid-takeover): the two most common tapes (42 and 39 copies) are mirror images of each other.
  Each executes for the full 8192-step budget, leaves itself intact, and writes a near-copy of its reversal into the partner; detector score 63/64.
  Zero bytes have vanished from the soup (fraction 0.000 from 0.004), `]` is rare (0.001) while `[` is at 0.118, `.` at 0.031, `>` at 0.085, `{` at 0.065.
- Epoch 163,840 (hoe 4.51, 5141 unique tapes): the dominant pair (55 + 55 copies) writes its *exact* reversal, score 64/64.
  As the second half of a pair with a random first half it loses ~20 bytes in the middle but keeps its two ends, which is why the two orientations persist.
- Mechanism as in the paper's Figure 4: read head walks one way, write head the other, one `[ ... ]` loop, so every generation flips orientation and a palindromic pair is the fixed point.
- The transient of seed 4 (peak brotli drop at epoch ~109,185) fell between checkpoints (106,496 and 110,592); by 110,592 every tape was unique again.
  Its soup was zero-enriched (13% zero bytes vs 0.4% at random) and `<`-rich (4.4%), i.e. a soup whose byte distribution had drifted, as the 2026 paper's Figure 2 describes.

## The transient of seed 4, scored (checkpoint 110,592, after the collapse)

- 11 of 8192 programs still score ≥ 5 on the detector, 6 of them ≥ 48 (scores 60–64): the lineage survived the collapse at low copy number.
- It is a different mechanism from seed 1's: a `[,}...},[` loop, i.e. it copies with `,` (tape[head0] = tape[head1]) while `}` walks the write head, and it keeps zero bytes around it rather than eliminating them.
- The soup at the collapse is zero-rich (13%) and every tape is unique: whatever spread from this lineage was not a faithful copy.
  Hypothesis to test: this replicator copies a window shifted relative to the 64-byte frame, so its children drift out of alignment and stop scoring, which the detector's fixed-position comparison would show as a collapse while the mechanism persists.

## The seed-1 collapse, scored (checkpoints every 4096 epochs, detector threshold 48)

| Epoch | hoe | Replicators (score ≥ 48) | Note |
| --- | --- | --- | --- |
| 172,032 | 5.66 | 5,495 / 8,192 | healthy: two-thirds of the soup replicates, every one scoring 63–64, a family of many variants |
| 176,128 | 3.18 | 106 / 8,192 | population down 50× in 4,096 epochs; the other 8,086 tapes score exactly 0 yet the soup is still highly compressible, i.e. it is full of near-copies that no longer replicate |
| 180,224 | 0.04 | 0 / 8,192 | extinct: not one tape scores above 0; the soup is random again |
| 184,320 | 3.19 | 2,294 / 8,192 | **a new lineage**: a palindromic copier built on `,` `}` `]` (`~y[[ < zYYY,= }] rMrq ] } } ] qrMr ]} =,YYYz < [[y~`), unrelated to the `.` `{` family that died; 28% of the soup 4,096 epochs after extinction |

So the replicators were not outcompeted by another replicator and did not drift out of frame: they were turned, in place, into non-functional near-copies, and the dead copies then scrambled each other into randomness within ~1,000 epochs (far faster than mutation alone, which needs ~4,096 epochs per byte).
The candidate explanation is a high-fecundity mutant whose children are broken (a copier that damages what it copies), which sweeps a small well-mixed soup before selection can remove it; the rebound at 184k and the second collapse fit a recurrence of the same event.
The checkpoints are in `experiments/results/checkpoints/` for a proper analysis (pair the 172k replicators against the 176k dead copies and look for the destructive interaction).

The rebound is re-emergence, not recovery: the 184k replicator shares nothing with the 172k family.
Re-emergence took under 4,096 epochs where the first emergence took 130,000.
The difference is the soup: a collapsed soup is random-looking to the compressor but its byte distribution is the one the replicators left behind, rich in instructions and poor in zeros, and the 2026 paper measures such distributions finding replicators 3 to 300 times faster than uniform bytes.
That makes the small-soup dynamics a cycle: emergence, takeover, collapse, fast re-emergence from the enriched remains.
Seed 4's second takeover at 337k fits the same pattern.
Detector outputs for all scored checkpoints are in `experiments/results/analysis_*.txt`.
