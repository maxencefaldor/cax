# Priors — every opinion in sampling and mutation

## Governing principle (user, 2026-08-30)

The sampling prior should be **as high-entropy as possible, spanning the full range of
life in Lenia**. Priority order:

1. **Width first** — prefer too wide over too narrow; a range must never exclude life
   (the catalogued envelope is a hard lower bound on width, not a target).
2. **Tighten second** — too wide means most samples are uninteresting; narrow only where
   doing so provably loses no life.

Decision metric: not viability % alone (a tightening metric), but the **diversity of the
viable set** — Vendi score and spread of viable random samples in the reference space.
Widen whenever the viable-set diversity does not drop; tighten only when it provably
does not.

No prior is unopinionated; the policy is: **as few opinions as possible, and every
remaining opinion either verified analytically (must contain all catalogued creatures)
or proven by experiment (viable-set diversity, then full QD outcomes)**. This file is
the exhaustive inventory. Verification tooling:
`check_priors.py` (deleted 2026-08-31, in git history: catalogued-envelope check + measured
viability and viable-set diversity of random samples).

## Sampling (`lenia.sample`, per strategy in `lenia/init.py`)

| # | Opinion | Status |
|---|---------|--------|
| P1 | `kernel_r_range (0.2, 1.0)` — uniform. Catalogued creatures use [0.5, 1.0]; the lower bound 0.2 gives a 2.4 px kernel at state_scale 1, seemingly too coarse to resolve the ring core. | **Measured 2026-08-30** (2048 noise-init samples per variant): viability 3.3% at low 0.2, 2.9% at 0.3, 2.6% at 0.4, 1.9% at 0.5 — monotonically *worse* when tightened, refuting the analytical suspicion. Widened to (0.1, 1.2) 2026-08-30 (measured free, user-approved); the ≲4 px low tail is grid-unresolvable at state_scale 1, revisit with full-QD diversity evidence. |
| P2 | `growth_mean_range (0.05, 0.5)` — uniform. Catalogued envelope [0.114, 0.489]: contained, upper margin only 0.011. | Analytical OK; tight upper margin noted. |
| P3 | `growth_std_range (0.005, 0.2)` — uniform. Was (0.005, 0.1), which **excluded** the Aquarium's 0.179 — exactly the feared single-channel-tuned range; caught and fixed 2026-08-30. | Fixed; analytical OK. |
| P4 | Weights ~ Dirichlet(1) on the simplex (uniform on the simplex). | **Paper-grounded (Chan 2020)**: the update uses h_k/h explicitly, so only the simplex direction is dynamical — total h is a pure time rescale. Sampling on the simplex is the exact non-redundant parameterization; Dirichlet(1) is max-entropy over it. |
| P5 | Ring rank ~ uniform{1..3}; ring heights uniform [0,1] with one ring pinned at 1.0. | **Paper-grounded (Chan 2019 §2.1.4)**: β is scale-invariant under kernel normalization, and Chan's canonical form is exactly max βᵢ = 1; uniform over the β hypercube is max-entropy over canonical kernels. Note the paper's rank-padding equivalence (β=(1) ≡ (1,0,0) with R×3): rank overlaps with r — mild redundancy, harmless. |
| P6 | Noise seed: uniform [0,1] blob of `blob_size (24)` px. | Opinion (size, distribution); untested. |
| P7 | `soliton_full`: catalogued rules + pattern, weights normalized to simplex (neutral under the update's normalization). | Analytical: exact official values. |

## Mutation (`lenia.mutate`)

| # | Opinion | Status |
|---|---------|--------|
| M1 | Bounded Gaussian, `mutation_std (0.01)` relative to each range, reflective boundaries. | σ untested (official iso 0.005 on raw units); reflection vs clip untested. |
| M2 | Weights: Dirichlet resampling, `weight_concentration (100)`. α→0 components can produce NaN child individuals (discarded as invalid). | Untested; consider α floor. |
| M3 | Ring heights: Gaussian + reflect at 0 + renormalize max to 1. Keeps Chan's max-1 convention. | Convention-matching; renormalization is opinion. |
| M4 | Ring rank is frozen at sampling (NaN pattern never mutates) — no ring birth/death. | Opinion by omission; official also froze structure. |
| M5 | Seed state: per-pixel bounded Gaussian. | Untested (official did the same). |
| M6 | No crossover (official used isoline). | Ledger V1 — untested. |

## Widening measurements (2026-08-30, 2048 noise-init samples per variant, 128² s1)

| Variant | Viable | Viable-set Vendi | Viable-set spread |
|---------|--------|------------------|-------------------|
| current ranges | 3.3% | 3.79 | 0.592 |
| growth_mean → (0.05, **0.7**) | **5.3%** | 3.80 | 0.515 |
| growth_std → (0.001, 0.3) | 1.2% | 3.50 | 0.759 |
| kernel_r → (**0.1, 1.2**) | 3.4% | **3.97** | 0.549 |
| all three widened | **5.7%** | 3.74 | **0.755** |

Readings under the width-first rule:
- **growth_mean upper 0.5 → 0.7**: 60% more life, species held — the thin 0.011 margin
  was hiding life above 0.5. Clear widen — **user-approved and applied 2026-08-30** (defaults + configs).
- **kernel_r → (0.1, 1.2)**: same viability, slightly more species — width for free.
  Clear widen — **user-approved and applied 2026-08-30** (defaults + configs).
- **growth_std → (0.001, 0.3)**: viability collapses to 1.2% and species drop, though
  the few survivors are unusually varied; 25 survivors is too few to judge — remeasure
  at larger n / intermediate width before deciding.
- Combined widening is net-positive (most life, broadest life, species held).
- **Applied combination verified 2026-08-30** (growth_mean + kernel_r widened,
  growth_std unchanged; 4096 samples): viability **5.0%** (was 3.3%), viable-set
  Vendi 3.66, spread 0.528 — consistent with the single-change measurements.

Caveats: prior-level metrics only (decisive test is full QD); Vendi across viable sets
of different sizes is roughly, not perfectly, comparable.

## Verification protocol

1. **Analytical (necessary condition)**: every range must contain the full catalogued
   creature envelope with margin. Resurrect `check_priors.py` from git history after any range change.
2. **Prior viability (cheap)**: fraction of random noise-init genotypes that develop
   into valid creatures — compare range variants on this number before any full run.
3. **Full QD (decisive)**: multi-seed runs judged on the diversity battery.
