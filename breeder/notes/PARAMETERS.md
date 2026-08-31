# Parameters — the evidence behind the numbers

The *values* live in the config models; this file holds only what the code cannot say:
**why** a number is what it is, and what would be lost by changing it.

Policy: **semantic parameters are never changed silently.** Every change gets a row here
(old → new, reason, evidence) and a line in the report to the user. Execution-only knobs
(`minibatch_size`, log/checkpoint intervals, `num_thumbnails`) are exempt.

Prior-setting principle (user, 2026-08-30): **width first.** Prefer too wide over too
narrow — a range must never exclude life, and the catalogued envelope is a hard lower
bound on width, not a target. Tighten only where it provably loses no life. The decision
metric is the *diversity of the viable set*, not the viability fraction, which is a
tightening metric. No prior is unopinionated; the policy is as few opinions as possible,
each either verified analytically (must contain every catalogued creature) or proven by
experiment.

## Genotype space

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| `kernel_r_range` | (0.1, 1.2) | Widened from the draft's (0.2, 1.0). Measured: tightening the lower bound monotonically *reduced* viability (3.3% → 1.9%), refuting the suspicion that sub-4px kernels are wasted; at (0.1, 1.2) viability held (3.4%) with slightly more species. **User-approved 2026-08-30.** Caveat open: 0.1·R·state_scale = 1.2 px is grid-unresolvable at scale 1. |
| `growth_mean_range` | (0.05, 0.7) | Widened from (0.05, 0.5). Measured: **60% more life** (3.3% → 5.3%), species held. The catalogued envelope's upper margin was only 0.011 — it was hiding life above 0.5. **User-approved 2026-08-30.** |
| `growth_std_range` | (0.005, 0.2) | The draft's (0.005, 0.1) **excluded** the Aquarium's own catalogued 0.179 — an envelope-check failure, exactly the feared single-channel-tuned range. |
| `min_mass` / `min_concentration` | 0.5 / 0.5 | `min_mass` is insensitive (dead worlds → 0, life ≥ 1.8, wide gap). `min_concentration` was judged visually: 0.5 right, 0.4 acceptable, 0.1 too low. |
| `spatial_dims` / `state_scale` | (128, 128) / 1 | Maximizes the world/kernel ratio, which the official code keeps at ~10.7. The draft's 64² at scale 2 is only 2.7 kernel radii wide — creatures are near world-sized and torus self-interaction is likely. Measured: 128² dominates on best and mean fitness, and scale 1 beat scale 2 by ~2× in honest units (`figures/world_scale.png`). |
| `R`, `T`, `state_scale` | frozen, not evolved | Discretization parameters, phenotype-neutral in the continuum limit — evolving them would be search noise. `T = 1/3` is not the catalogued `T`: CAX uses weight-averaged growth, so `T` absorbs the family's Σh (Aquarium Σh ≈ 6, official `T=2` ≡ CAX `T=1/3`, verified equivalent). |
| kernel topology | fixed 9 self + 6 cross | Same shape as the official code. Evolvable topology deferred. |
| weights | Dirichlet on the simplex | Replaces the official raw growth heights `h`, which carry an overall-intensity degree of freedom that is a pure time rescale in the continuum limit — a neutral search dimension. |

## Mutation operators

Defaults are the draft's; the arms are opt-in and under investigation (2026-08-31).
Evidence and the measured pathologies are in EXPERIMENTS.md.

| Parameter | Default | Arm | Why |
|-----------|---------|-----|-----|
| `mutate.weight_floor` | 0 | 0.5 | Pure `Dir(c·w)` drift is absorbing at the simplex corners: 6 of 8 rules dead by ~100 applications. A Jeffreys pseudo-count leaves zero dead after 1000. Grid puts the knee at 0.5 (0.1 under-protects, 1.0 only adds pull toward uniform). |
| `mutate.weight_concentration` | 100 | 1600 | 100 is the draft's untuned value; measured, it gives weights 27% relative steps while every other parameter takes ~1% of its range. 1600 restores parity. |
| `mutate.state_strategy` | `gaussian` | `frozen`, `multiplicative` | Reflection at 0 makes empty pixels gain mass every mutation. `multiplicative` is support-preserving. `frozen` asks whether seed evolution (49k of ~49k+30 dimensions) earns anything. |
| `mutate.mutation_std` | 0.01 | — | Relative to each parameter's range. Unchanged from the draft. |

**Divergence from notebook 64, deliberate**: beta is renormalized **per kernel**
(`nanmax(beta, axis=-1)`). The notebook takes the max across all kernels, which breaks
the peak-1 invariant its own sampler establishes.

## Fitness and descriptor

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| fitness definition | `norm_mean velocity` | **User-directed 2026-08-30**: spinners hack instantaneous speed (center-of-mass tangents score without travel); the norm of the time-mean velocity cancels rotation (verified: spinner 0.05 → 0.000, straight traveler unchanged). Orbits longer than the window can still score partially. Best-fitness values are **not comparable across fitness definitions**. |
| `fitness.window` | 32 | Excludes the developmental transient (official `n_keep` behavior). Untested as an isolated ablation. |
| metrics unit | `R × state_scale` | Bug fix to official semantics. Pre-fix, scale-2 runs had velocity ×2 and mass ×4. |
| descriptor from | encoder **means** | Deterministic; removes sampling noise from the archive geometry. |
| homeostasis / polarization | available, not recommended as objectives | Both saturate (exactly 0.0000 / 1.0), after which diversity alone drives the run. Better as descriptor axes. |

## Encoder

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| `augment` | `[d4, roll]` | The dynamics' own symmetry group — exact-on-grid transforms only, no photographic policies. Extends the official flip+rot90. **Judged by eye 2026-08-31**; roll beats no-roll on both seeds. |
| `invariant` | `true` | `encode` averages features over the D4 orbit, making the series *exactly* flip/rot90-invariant (verified allclose 1e-8). Augmentation only discourages pose-farming; orbit averaging closes it mathematically. 8× encoder forward cost — negligible for the VAE, significant for VGG. |
| lr / grad clip / steps | 1e-3 constant / 1.0 / 8192 | 24-config grid on 2048 curated creatures plus a live-run probe: the draft's 4096 steps were systematically under-trained. 16384 buys ~3.4% more reconstruction, not visible in the search. |
| `features` | (3, 16, 32, 32) | The extra conv keeps the encoder width at 8192 on 128² frames; without it the linear layer is ~1B parameters. |
| training data | valid individuals only | Placeholders and invalid individuals never train the descriptor. |
| `latent_size` | 8 | **Open** — the 5-seed verdict was decided on Vendi in the random reference space, since discredited. See EXPERIMENTS.md. |

## Reference space (the measuring instrument)

Not search parameters, but they decide which runs *look* better on paper.

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| reference encoder | frozen pretrained VGG16, layer 11 | Replaced a frozen random CNN 2026-08-31. Validated against the eye's roll/no-roll verdict: every VGG measure agrees on both seeds, every random-space measure fails one. **Breaks comparability with all diversity/vendi logged before 2026-08-31.** |
| crop / resize | 4 physical units, then 64² | The crop fixes the *physical* window so a creature embeds the same in any world size; the resize fixes the *pixel* grid so one space serves all systems. |
| `vendi` kernel | covariance | The correlation kernel discards how far an individual sits from the mean, so a collapsed population's near-orthogonal residual noise scored near maximum. Old kernel still logged as `vendi_cos`. |
| batch | 128 | VGG activations are ~100× the frames that produce them; one call over 1024 individuals peaks at ~5 GiB. |

## Particle Life

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| `num_steps` | 1024 | Measured relaxation: the seeded blob settles by ~1024 steps. At 256 the search would select on the transient. |
| `num_frames` | 64 | Rendering one frame costs about as much as a simulation step, so only the observed tail is drawn. |
| `num_particles` / `num_classes` | 256 / 4 | Cost is quadratic in particles; 4 classes is the classic setting. |
| `r_max`, `dt`, `force_factor`, `velocity_half_life` | 0.15, 0.01, 1.0, 0.01 | Frozen. `r_max` is the length unit every metric is expressed in — evolving it would make measurements incomparable, the same reason Lenia freezes `R`. |
| `sample.strategy` | `blob` | **55% viable from a blob vs 0% from a uniform scatter.** A soup essentially never self-organizes into a soliton within budget. |
| gates | concentration 0.5, clustering 2.0 | Particles never die, so the gate must catch loss of *organization*, on the two axes a soup and a soliton differ on independently. Keeps 31% of the prior. |
| positions mutate | wrapping, not reflecting | They live on a torus. |

## Channel-size transfer (verified in code, 2026-08-30)

Which parameters survive a change of `channel_size`:

- **Transfer exactly** — `growth_mean_range`, `growth_std_range` (each kernel reads one
  source channel via one-hot and its kernel sums to 1, so U ∈ [0,1] for any channel
  count); `kernel_r_range`, `beta`, `kernel_rank` (pure kernel geometry); the weight
  simplex; state/blob ranges; all mutation scales (relative to ranges); descriptor
  inputs (render always emits `(H, W, 3)`).
- **Must be re-set** — `T` (absorbs the family's Σh: 3-channel Aquarium 1/3, 1-channel
  Orbium ≈ 10) and `R` likewise; `min_mass` (mass sums over channels — measured exactly
  ×3 from c=1 to c=3, so 0.5 is a 3-channel calibration); `num_cross_kernels`
  (structural — no cross pairs at c=1); `pattern_names` (checked at load).
- **Transfers with a caveat** — `min_concentration` is channel-count-free but
  world-size-relative.
