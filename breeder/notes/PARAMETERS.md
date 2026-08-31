# Parameters — every change from the draft, and the current values

Policy: **semantic parameters are never changed silently.** Every change gets a row here
(old → new, reason, evidence) and a line in the report to the user. Execution-only knobs
(minibatch_size, log/checkpoint intervals, num_thumbnails) are exempt from the policy
but listed for completeness. "Default" = code default in `main.py`/`config.py`;
"Protocol" = the value in `breeder/configs/*.yaml` used by the current campaign.

## Changed since the draft notebook

| Parameter | Draft | Default | Protocol | Why / evidence |
|-----------|-------|---------|----------|----------------|
| `lenia.spatial_dims` | (64, 64) | (64, 64) | **(128, 128)** | 64² world is only 2.7 kernel radii wide (figure + ablation 1: 128² dominates on best and mean fitness). ⚠ default ≠ protocol. |
| `lenia.state_scale` | 2 | 2 | **1** | Maximize world/kernel ratio (official 10.7); s1 beat s2 (~2× in honest units). ⚠ default ≠ protocol. |
| `lenia.growth_std_range` | (0.005, 0.1) | (0.005, 0.1) | **(0.005, 0.2)** | Draft range **excluded** the Aquarium's catalogued 0.179 — envelope check failure, exactly the feared single-channel-tuned range. ⚠ default ≠ protocol. |
| `lenia.init` (was implicit) | noise blob | noise | **soliton_full (5N7KKM)** | User-directed protocol: breed a lineage from one full soliton (rules + pattern). |
| `lenia.num_frames` | 32 | — removed | — | Lenia made neutral: `develop` renders every step; consumers apply `window`. |
| metrics unit (internal) | R | R × state_scale | same | Bug fix to official semantics; mass/velocity now invariant to state_scale. Pre-fix scale-2 runs had velocity ×2, mass ×4. |
| `qd.num_init` (was init_scale×pop) | 8192 | 8192 | **1** | User-directed: population grows from a single living solution; invalid placeholders fill the rest. ⚠ default ≠ protocol. |
| `search.num_generations` | 256 | 256 | **1024** | Budget choice for the campaign. ⚠ default ≠ protocol. |
| fitness convention | minimize | **maximize** | — | Official/GA convention (pure sign flip, semantics identical). |
| `fitness.window` | all 128 steps | **32** | 32 | Excludes the developmental transient (official n_keep behavior). **Untested as an isolated ablation.** |
| `descriptor.latent_size` | 2 | **8** | 8 | Tested (5 seeds): latent 8 = most effective species (Vendi ≈ 8.0, σ 0.6); 2 was a regression. Ledger D1 closed. |
| `descriptor.features` | (3, 16, 32) | (3, 16, 32) | **(3, 16, 32, 32)** | Extra conv keeps the VAE encoder width at 8192 on 128² frames (else a 1B-param linear). ⚠ default ≠ protocol. |
| VAE training schedule | 64 epochs × full data, batch 32 | **4096 sampled steps, batch 256** | same | Official random-(individual, frame) scheme; single compiled scan. **Untested as an isolated ablation.** |
| VAE training data | all individuals | **valid individuals only** | same | Principled: placeholders/invalid never train the descriptor. |
| VAE retrain | fresh every 32 gen | same | same | Unchanged from draft (ledger D2 still open). |
| `lenia.growth_mean_range` | (0.05, 0.5) | **(0.05, 0.7)** | (0.05, 0.7) | Measured (2048 samples/variant): upper 0.5→0.7 gives 60% more life (3.3%→5.3%), species held — the 0.011 envelope margin was hiding life above 0.5. **User-approved 2026-08-30.** |
| `lenia.kernel_r_range` | (0.2, 1.0) | **(0.1, 1.2)** | (0.1, 1.2) | Measured: viability held (3.4%), slightly more species (Vendi 3.97) — width for free. Caveat stands: 0.1·R·state_scale = 1.2 px kernels are grid-unresolvable at scale 1, so the low tail spends samples on noise, but measured harmless. **User-approved 2026-08-30.** |
| fitness schema (agnostic refactor) | `name: speed` preset | `name: metric, metric: linear_velocity, reduce: mean, sign: 1` | same | Schema-only, semantics identical (speed ≡ mean linear_velocity): metric fitness/descriptor configs now name the system's metric series directly, validated against the complex system's declared metrics. Config sections: `lenia:` → `complex_system: {name: lenia, ...}`. |
| encoder architecture (schema v4) | descriptor variants own weights | named `encoders:` (vae/vgg) enrich `phenotype.series`; fitness/descriptor = weightless reductions `{series, reduce, ...}` / `[[series, scale], ...]` | same values | Schema-only, semantics identical for every current experiment (verified: reductions match hand computation; smoke over vae/metrics/deepdream/resume). `train_interval` is now top-level. Execution-level change: the archive (`observations`) spans the largest encoder window instead of all frames — metric-descriptor runs stop holding 128-frame archives (4× memory) for nothing. |

| fitness definition | mean instantaneous speed (`mean linear_velocity`) | **`norm_mean velocity`** (net displacement per unit time over the window) | same | **User-directed 2026-08-30**: spinners hack instantaneous speed (center-of-mass tangents score without travel); the norm of the time-mean velocity vector cancels rotation (verified numerically: spinner 0.05 → 0.000, straight traveler unchanged). Caveat: orbits longer than the 32-step window can still score partially. `linear_velocity` remains available and stays the metric-descriptor axis. |
| new experiment: homeostasis | — | — | `fitness: {series: latent, reduce: var, sign: -1}` | Ledger D4 restored: the paper's unsupervised fitness, now a config line under the encoder architecture. |
| `encoder.augment` **default** | no augmentation (draft) / flip+rot90 (official, continuous training) | `augment: [d4, roll]` opt-in on the VAE encoder: per-sample dihedral-group element + torus roll on each training batch | **`[d4, roll]`** (locked in 2026-08-31 after the user judged the wave sheet: orbit+augment arms produce coherent creatures, the no-augment base stays micro-dots) | Restores and extends official Leniabreeder's augmentation with the dynamics' own symmetry group only — exact-on-grid transforms, no photographic policies. Default off: every experiment that enables it is a visible config diff. Caveat: `roll` spans the full torus (uniform 0..127 per axis), not a small centering-residual jitter — the in-distribution variant remains untested. |
| `encoder.invariant` (new) | — (no counterpart in official code) | `invariant: true` on any encoder: `encode` averages features over the D4 orbit — the emitted series is *exactly* flip/rot90-invariant (verified: allclose 1e-8; orbit closure proven) | **`true`** (locked in 2026-08-31, same judgment) | Beyond-official lever: augmentation only discourages pose-farming, orbit averaging closes it mathematically. 8× encoder forward cost (negligible for the VAE, significant for VGG). Under ablation (base_aug_orbit, vgg_orbit), judged by the user's eye. |
| singular encoder (schema v6) | named `encoders:` dict (0..n) | `encoder:` — one `EncoderConfig` or `null` (no-op) | same values | Schema-only (**user-directed 2026-08-31**): no experiment ever needs two encoders now that deepdream (vae descriptor + vgg fitness) is retired. Each encoder type emits a fixed series name (`vae` → `latent`, `vgg` → `vgg`); the base `Encoder` is the no-op, so encoder-less runs share the code path. Deepdream experiments removed from the protocol. |
| lenia sample/mutate split (schema v5) | `complex_system: {init, blob_size, pattern_names, mutation_std, weight_concentration}` flat | `complex_system.sample: {strategy, blob_size, pattern_names}`, `complex_system.mutate: {mutation_std, weight_concentration}` | same values | Schema-only, semantics identical (**user-directed 2026-08-31**): a complex system samples or mutates — "init" is the QD algorithm's concept, so `init:` becomes `sample.strategy:`. Parameter ranges stay top-level: they define the genotype space, consumed by both operators. Old run dirs no longer `--resume` on new code (config.yaml schema). |

## Particle Life (new system, 2026-08-31)

| Parameter | Value | Why / evidence |
|-----------|-------|----------------|
| `num_particles` / `num_classes` | 256 / 4 | Cost is quadratic in particles; 4 classes is the classic setting. |
| `num_steps` | 1024 | Measured relaxation: the seeded blob settles by ~1024 steps (concentration 0.90 → 0.40, clustering 12.7 → 4.0, thermal velocity → 0.04). At 256 steps the search would select on the transient, not the creature. |
| `num_frames` | 64 | Rendering costs `resolution^2 * num_particles` per frame, on par with a simulation step, so only the observed tail is drawn. |
| `r_max`, `dt`, `force_factor`, `velocity_half_life` | 0.15, 0.01, 1.0, 0.01 | Frozen. `r_max` is the length unit every metric is expressed in — evolving it would make population measurements incomparable, the same reason Lenia freezes `R`. The rest are integration/time-scale parameters. |
| `beta_range` | (0.1, 0.7) | Evolved: the one dimensionless shape parameter of the force law. |
| `sample.strategy` | `blob` (radius 0.1) | Measured: 55% of random genotypes are viable from a blob, **0% from a uniform scatter** — a soup essentially never self-organizes into a localized soliton within the budget, so the localized seed is what makes the search possible. |
| `min_concentration` / `min_clustering` | 0.5 / 2.0 | Calibrated on 512 settled genotypes: keeps 31% (settled percentiles — concentration p50 0.36, clustering p50 4.0). Concentration is the binding gate at this setting; clustering guards the diffuse-but-centered case and is a descriptor axis in its own right. |
| `qd.num_init` | 1024 | No catalogued creature exists for these systems, so the protocol is de-novo discovery from a large sample, then downselection — the analogue of Lenia's single-ancestor seeding for a system that has an ancestor. |

## Unchanged from the draft (verified where noted)

`R 12`, `T 1/3`, `num_steps 128`, `channel_size 3`, kernel topology (9 self + 6 cross),
Gaussian ring core + exponential growth (verified = official), `blob_size 24`, `kernel_rank 3`,
`mutation_std 0.01`, `weight_concentration 100`, `min_mass 0.5` (calibrated: insensitive
in [0.5, 1.8]), `min_concentration 0.5` (**calibrated 2026-08-30, user-confirmed** via
blur ladder + gate-0.05 QD map + gate-0.1 A/B; 0.4 acceptable, 0.1 too low), `population_size 1024`, `num_children 256`,
`k 3`, `sample_ratio 0.0`, `descriptor.window 32` (draft's num_obs_steps), deterministic
encoder means.

## Execution-only (exempt, listed for completeness)

`minibatch_size 256` (32 for VGG runs — memory bound), `checkpoint_interval 32`,
`log_interval 8`, `num_thumbnails 1024`; diversity evaluation constants
(`REFERENCE_CROP_RADII 4`, reference encoder architecture, Vendi correlation kernel) —
these shape measurement, not search, and are frozen across all experiments.

## Channel-size transfer (verified in code, 2026-08-30)

Which ranges transfer unchanged to a different `channel_size` (e.g. 1), and which must
be re-set:

- **Transfer exactly** — `growth_mean_range`, `growth_std_range` (each kernel reads ONE
  source channel via one-hot and its kernel is normalized to sum 1, so U ∈ [0,1] for any
  channel count — `perceive.py`); `kernel_r_range`, `beta`, `kernel_rank` (pure kernel
  geometry); weight simplex (Chan 2020's h_k/h runs over all kernels for any c);
  state/blob ranges ([0,1] per channel by definition); all mutation scales (relative to
  ranges); descriptor inputs (render always emits (H, W, 3), zero-filled).
- **Must be re-set** —
  `T`: with CAX's normalized-weight convention, T absorbs the catalogued family's Σh
  (3-channel Aquarium: T = 2/6 = 1/3; 1-channel Orbium: h = 1, so T = the catalogued
  T ≈ 10). `R` likewise per family (Orbium 13).
  `min_mass`: mass sums over channels (measured: exactly ×3 from c=1 to c=3 at equal
  density), so 0.5 is a 3-channel calibration — scale ≈ ∝ channel_size, or (proposal,
  pending approval) redefine mass per channel to make the threshold transfer.
  `num_cross_kernels`: structural — no cross pairs at c=1.
  `pattern_names`: soliton inits must use patterns of matching channel count (checked
  at load).
- **Transfers with a caveat** — `min_concentration` (normalized by total mass, hence
  channel-count-free, but world-size-relative as already noted).
- Code fix: c=1 previously produced a float32 kernel topology (empty cross list
  promoted the dtype); now explicit int32.

## Mutation operators (2026-08-31, user-directed after audit)

Two measured pathologies of the draft operators, each now an opt-in arm varied alone
against base (defaults unchanged: `weight_floor: 0`, `state_strategy: gaussian`).

| Parameter | Draft | Arm | Why / evidence |
|-----------|-------|-----|----------------|
| `mutate.weight_concentration` | 100 (notebook 64's value, never tuned) | 1600 in the `weight_tuned` arm | Characterized on CPU: at c=100 a uniform weight takes 27% relative steps per mutation — the largest steps in the genotype, against the doctrine that every parameter moves ~1% of its range; c=1600 restores parity (std 0.0084 ≈ 0.01 absolute). Big steps also accelerate fixation. Floor grid: 0.1 leaves ~1 dead rule per lineage at 1000 applications, 0.5 leaves zero at every c, 1.0 only adds pull toward uniform — 0.5 is the measured knee, kept. |
| `mutate.weight_floor` | 0 (pure `Dir(100·w)`) | 0.5 | Simulated 256 lineages of iterated mutation: pure drift kills 6 of 8 rules (< 1e-6) by ~100 applications and all but one by ~300 — inside one 1024-generation run; a weight near 0 gets alpha near 0, which is absorbing. Floor 0.5 (Jeffreys-style pseudo-count): zero dead rules after 1000 applications, mean max weight 0.34, simplex closure kept. |
| `mutate.state_strategy` | `gaussian` (bounded per-pixel, reflect at 0) | `frozen`, `multiplicative` | Reflection at 0 makes every empty pixel *gain* `sigma*sqrt(2/pi) ~ 0.008` mass per mutation — iterated, background pixels random-walk away from 0 and validity gates clean up after the operator. `multiplicative` (`x * exp(sigma*noise)`, reflect at 1) is support-preserving: empty stays exactly empty (verified). `frozen` asks whether seed evolution earns anything at all: the seed is 49k of the genotype's ~49k+30 dimensions and development is an attractor. Notebook-64 heritage: same audit found its beta normalization takes the max across all kernels, breaking its own sampler's per-kernel peak-1 invariant — breeder normalizes per kernel. |

## Reference space (2026-08-31, user-directed)

The measuring instruments behind the logged `diversity`/`vendi` columns — not search
parameters, but they decide which runs *look* better on paper, so they get rows.

| Parameter | Before | After | Why / evidence |
|-----------|--------|-------|----------------|
| reference encoder | frozen *random* CNN (3→32→64→128, fixed rng, never trained) | frozen *pretrained* VGG16 up to layer 11 (third block, 256 ch), same physical crop (4 units) + 64² resize, chunked at 128 | Scored all four roll/no-roll populations in both spaces against the user's visual verdict (roll > no-roll, both seeds): every VGG measure agrees on both seeds, every random-space measure fails at least one. Random projections preserve *pixel* distance, and near-identical creatures differing in phase/micro-detail are far apart in pixel space; perceptual features collapse exactly those nuisance directions (the LPIPS argument). **Breaks comparability with all previously logged diversity/vendi values.** |
| `vendi` kernel | correlation (centered features rescaled to unit length) | covariance (centered, unit trace) | The unit-rescaling discards how far each individual sits from the population mean; a collapsed population's residuals are near-pure noise, and normalized high-dim noise is near-orthogonal — so collapse scored near maximum (it ranked the monotonous no-roll run 6.09 vs the varied roll run 4.76). Covariance kernel separates them 1.6–2.1× in VGG space, the cleanest of all measures tested. The old kernel stays logged as `vendi_cos` so future disagreements stay visible. |

## Defaults aligned with the protocol (2026-08-31, user-directed)

The converged protocol is now the code default, so a config states only its delta and a
bare `python -m breeder.main` runs the protocol. **Verified faithful**: every surviving
regenerated config is byte-identical to its committed predecessor except the two rows
below. Settled one-off experiment configs (60 → 15) are deleted; definitions live in git
history, conclusions in EXPERIMENTS.md.

| Default | Before | After (= protocol) |
|---------|--------|--------------------|
| `lenia.spatial_dims` / `state_scale` | (64, 64) / 2 | (128, 128) / 1 |
| `lenia.sample.strategy` / `pattern_names` | noise / (5N7KKM, VT049W) | soliton_full / (5N7KKM,) |
| `lenia.growth_std_range` | (0.005, 0.1) | (0.005, 0.2) — the Aquarium's catalogued stds reach 0.18 |
| `flow_lenia.sample.strategy` | inherited Lenia's | noise (own default: catalogued solitons are not its solitons) |
| `qd.num_init` / `num_generations` | 8192 / 256 | 1 / 1024 |
| `encoder.features` (VAE) | (3, 16, 32) | (3, 16, 32, 32) — the 128² conv stack |

Semantic changes hidden in the regeneration, called out per the doctrine:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `vgg` encoder `invariant` default | `false` | `true` | Adopts the `vgg_orbit` arm as the texture descriptor's default, matching the VAE's locked-in orbit invariance; only affects `vgg_descriptor` runs. |
| `flow_lenia` `pattern_names` | (5N7KKM, VT049W) | (5N7KKM,) | Inert: its strategy is `noise`, patterns unused. |
