# Server plan (8 × H100)

What to run first when the branch lands on the GPU box, in priority order.
Every script takes a seed and writes CSV or npz under an output directory; one process per GPU (`CUDA_VISIBLE_DEVICES=k`) is the intended parallelism, no sharding.

## Setup

```sh
git clone -b bff https://github.com/maxencefaldor/cax && cd cax
uv sync --all-extras --dev          # then install the CUDA jax wheel for the box
uv run pytest tests/test_cs/test_bff.py -q
```

Optional but valuable: build cubff for CPU (`clang++`/`g++`, `libbrotli-dev`; see `notes/experiments/harness.cc`) and rerun `fidelity.py`, `replay.py` and `detector_fidelity.py` on the new machine.
They pin exactness on that JAX build too.

## 1. Phase 1 at full scale (closes the caveat)

`notes/experiments/soup_run.py OUT --seed S --num_programs 131072 --max_epochs 16384 --stop_at 1.0 --stop_after 4096`

8 seeds in parallel, 16k epochs each.
Expect a few seconds per epoch pre-transition on an H100 (≈ 20 ms per 2^17-pair epoch on the CPU-exact path is not achievable; measure the first 64 epochs and report).
Deliverable: fraction of runs with high-order entropy ≥ 1 within 16k epochs (paper: 40%), and whether full-size takeovers persist (they did not at 8192 programs).

## 2. Phase 2 appearance rates (the instruction-set comparison)

`notes/experiments/appearance_rate.py OUT.csv --num 1000000 --batch 8192 --controls matched,cyclic,flip --heads two,swap --dists uniform,cust`

One combination per GPU.
The 20k-program CPU run gives the first numbers; 10^6 per combination is the target to resolve rates near 10^-5.

## 3. Minimal replicators per variant

`notes/experiments/minimal_replicator.py OUT.csv --length L --control C --dialect D [--with_zero]`

Exhaustive for L ≤ 6 (10^6 programs at 10 symbols); sample for L = 7, 8.
Then `robustness.py --from_csv OUT.csv` on the hits.

## 4. Variant soups

`soup_run.py OUT --control flip --dialect two --num_programs 131072 --max_epochs 16384` and the seeded version with a hand-written variant replicator once one is known from step 3.
Total variants never halt, so every pair runs 8192 steps: expect ≈ 10× the epoch cost of `matched` on a random soup.

## What to look at first in the results

- Appearance rate per variant vs `matched`: if `flip` is within a factor of a few, Phase 3 is justified; if it is 100× rarer, the 2D design needs a different loop primitive.
- Whether full-size `matched` takeovers persist.
  If they collapse at 2^17 too, the paper's Figure 5 needs re-reading.
- Replicator length distribution per variant, not just the minimum.
