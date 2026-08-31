# Modification ledger — modern Leniabreeder vs. official code

Official reference: [maxencefaldor/Leniabreeder](https://github.com/maxencefaldor/Leniabreeder)
([arXiv:2406.04235](https://arxiv.org/abs/2406.04235), ALIFE 2024).
Current draft: `cax/examples/64_leniabreeder.ipynb` on top of `cax.cs.lenia`.

Every divergence from the official code gets a row. Columns:

- **Principled** — Yes / No / Maybe: is the change justified from first principles,
  independent of empirical outcome?
- **Tested** — No / Partial / Yes: has an experiment isolated this change?
- **Improvement** — ? / Yes / Neutral / No: empirical verdict once tested.
- **Conf** — 1–10: confidence in the current overall judgment (principled + verdict).

| ID | Modification (official → draft) | Principled | Tested | Improvement | Conf |
|----|--------------------------------|------------|--------|-------------|------|
| S1 | Simulation: bespoke `lenia.py` → CAX `Lenia` (nnx, perceive/update, N-D). Conventions verified equivalent: official `T=2` with raw-`h` growth ≡ CAX `T=1/3` with weight-averaged growth (Σh≈6 for Aquarium); example 20 reproduces official solitons. | Yes | Partial | Neutral | 9 |
| S2 | World 128², 200 steps → 64², 128 steps with `state_scale=2`. Effective kernel radius is R·state_scale = 24 px in a 64-px world (world/kernel ≈ 2.7 vs ≈ 10 officially) — creatures are near world-sized, torus self-interaction likely. | No | No | ? | 7 |
| S3 | Kernel core: Gaussian ring core, std 0.15 — **identical** to official (`bell(D%1, 0.5, 0.15)`); this is the standard multi-channel/expanded-universe core (Chan 2020), kept fixed. Growth `2·bell(u,m,s)−1` identical. Not a divergence; recorded to keep it frozen. | Yes | — | Neutral | 10 |
| S4 | Per-step world re-centering + in-step stats → post-hoc toroidal circular statistics (`metrics_fn`). | Yes | No | ? | 8 |
| S5 | Metrics unit fix: `develop` passed raw `R` to the metrics, but the official code uses `R × world_scale` as the physical unit. Fixed 2026-08-30 — mass and velocity are now invariant to `state_scale` and validity thresholds transfer across world configs. Runs before the fix had scale-2 velocities inflated 2× and masses 4×. | Yes | — | Yes | 9 |
| G1 | Genotype: flat 3117-vector → structured pytree dataclass. | Yes | No | ? | 9 |
| G2 | Kernel radius `r` and ring heights `beta` (variable rank 1–3, NaN-masked) evolved (official: frozen from pattern). Expands rule search space. | Yes | No | ? | 8 |
| G3 | Raw growth heights `h` → Dirichlet weights on the simplex. Removes the overall-intensity degree of freedom, which in the continuum limit is a pure time rescale (neutral/search-noise dimension). | Yes | No | ? | 8 |
| G4 | Init from Aquarium soliton + noise → uniform random rules + random 24×24×3 seed blob (8× oversampled initial population). De-novo discovery instead of breeding one lineage. To become a configurable *seeding strategy* (solitons / random / learned / mixed). | Yes | No | ? | 6 |
| G5 | Explicit per-parameter ranges with reflective bounds (official: unbounded floats). | Yes | No | ? | 8 |
| G6 | Kernel topology fixed 9 self + 6 cross (same shape as official, now constructed). Evolvable topology deferred — not a priority. | Yes | — | Neutral | 8 |
| G7 | R, T, state_scale fixed and **not** evolved: discretization/scale parameters, phenotype-neutral in the continuum limit (search noise). | Yes | — | Neutral | 8 |
| V1 | qdax isoline (iso 0.005, line 0.05, crossover-like) → structured per-parameter mutation: bounded Gaussian + reflection; Dirichlet resampling (conc. 100) for weights; reflect+max-normalize for beta. No crossover. | Yes | No | ? | 6 |
| A1 | CVT MAP-Elites / AURORA unstructured repertoire → Dominated Novelty Search (population 1024, µ+λ, k=3 nearest fitter neighbors). No grid, no descriptor bounds, no distance threshold. | Yes | No | ? | 7 |
| A2 | Fitness maximized/−inf → minimized/+inf. **Reverted**: `breeder/` maximizes, matching the official code and GA convention. | Yes | — | Neutral | 9 |
| E1 | Failure flags (`is_empty`/`is_full`/`is_spread`) → toroidal degeneracy: any-step mass < 0.5 or circular concentration < 0.5. **Calibrated 2026-08-30**: min_mass insensitive (dead worlds → 0, life ≥ 1.8, wide gap); min_concentration judged visually via the blur ladder, a gate-0.05 QD map, and a gate-0.1 A/B — user verdict: **0.5 right (0.4 acceptable), 0.1 too low**. The +68% Vendi at gate 0.1 therefore counts non-life: a caution about reference-space Vendi vs human judgment. | Yes | Yes | Neutral | 8 |
| E2 | Fitness hardcoded to mean instantaneous speed over **all** steps (official: configurable metric over last `n_keep` steps, excluding transient). **Fixed in `breeder/`**: `MetricFitness(metric, reduce, window, sign)` restores configurability and transient exclusion. | Yes | No | ? | 8 |
| E3 | Phenotype 32×32 raw-state crop each step → full-world 64×64 RGB uint8 renders of last 32 steps, centered post-hoc. uint8 saves memory; render ≡ raw state for C=3. | Yes | No | ? | 7 |
| D1 | VAE: linen, features 128, latent 6–8 → nnx `cax.nn.vae`, features (3,16,32), latent 2. **Tested (ablation 2, single-ancestor protocol, multi-seed)**: latent 8 holds the most effective species (Vendi ≈ 7.4–8.5 across 3 seeds) vs latent 2 (6.3), 16 (4.7–6.9), 32 (4.8–5.0); fitness is seed-noise-equivalent across latents. Verdict: keep latent 8 — the draft's latent 2 was a regression, the paper's 8 confirmed. | Yes | Yes | Yes | 8 |
| D2 | Continuous training (8 epochs/gen, LR schedule, grad clip, flip/rot90 augmentation) → full retrain from scratch every 32 generations, plain Adam, no augmentation. `breeder/` keeps retrain-from-scratch but adopts the official random-(individual, frame) batch sampling for a fixed step count. Paper lists missing rotation invariance as a limitation. **Augmentation restored 2026-08-31** as opt-in `encoder.augment: [d4, roll]` (symmetry group only, beyond official's flip/rot90) plus `encoder.invariant` (exact D4 orbit-averaged encoding, no official counterpart) — under ablation, user's eye judges. Retrain-from-scratch itself still untested vs continuous. | Maybe | Partial | ? | 5 |
| D3 | Descriptor from **sampled** latents (stochastic) → encoder **means** (deterministic). Removes noise from the archive geometry. | Yes | No | ? | 8 |
| D4 | Unsupervised homeostasis fitness (latent variance) dropped in draft. To be restored as one option of the configurable fitness. | No | No | ? | 7 |
| X1 | Hydra + vendored qdax + pickle scripts → self-contained notebook → **`breeder/` package** (`core/` + `lenia/`, pydantic config, orbax checkpoints, `python -m breeder.main`), notebook demoted to demo. | Yes | Partial | ? | 9 |
| X2 | Resume bug fix (2026-08-30): the orbax restore template (`abstract_fresh_state`) skipped `init_population`'s padding, so resuming any run with `num_init < population_size` (the single-ancestor protocol!) crashed or mismatched. Only `num_init ≥ population_size` runs ever resumed. Padding is now shared (`pad_population`) between the real and abstract init paths; caught by smoke12's `num_init: 2` resume leg. | Yes | Yes | Yes | 9 |
| X3 | Resume removed (2026-08-31, user-directed): `--resume`, `checkpoint.restore`, and the abstract restore template deleted — runs are cheap and fast, and the resume path (X2) was the most fragile code in `main`. Checkpoints are still written every `checkpoint_interval`, so resuming can be rebuilt if ever needed. | Yes | — | Neutral | 9 |

## Planned (not yet implemented)

- Configurable fitness and descriptor protocols (metric-based, VAE/AURORA, fixed CNN à la
  example 46, unsupervised homeostasis) — user requirement.
- Configurable seeding (soliton set / random blob / learned init) and emitter mix
  (fresh-sample vs. mutation ratio; 100% sample = pure random search).
- Complex-system-agnostic search core (`breeder/core/`) with one package per complex
  system under `breeder/cs/` (`lenia/`, `flow_lenia/`, `life/`).
- Gene-filter–based choice of which parameters QD optimizes (nnx filters).
- Multi-GPU experiment runner: one experiment per GPU (or fractions) for ablation grids;
  mesh + `device_put` + jit sharding for single large runs.

## Structure (2026-08-31, user-directed review then "fix 1, 2, 3")

- **report is a package**: `breeder/report/` = `__init__.py` (data assembly) +
  `template.html` + `style.css` + `script.js`. The page was a 500-line Python string —
  unlintable, unformattable; the video-decoder LRU bug shipped through that blind spot.
- **Lenia-family coupling made explicit**: `observe` is exported from `lenia`'s public
  API and `flow_lenia` imports it from the package, not the submodule. Flow Lenia *is*
  Lenia with a mass-conserving update: config subclasses `LeniaConfig`, genotype half
  re-exported — the base relationship is by design, now stated.
- **`ComplexSystem` callables renamed** `sample_fn/mutate_fn/develop_fn/valid_fn` per
  the DESIGN.md `*_fn` convention (`valid` collided with the boolean mask, `sample`
  with `SampleConfig`).
- **`reference_features` owns the physical crop**: the `round(RADII * unit)` arithmetic
  was duplicated in main.py and report; now one function in `diversity`.
- **`cs/` package**: the three complex systems move under `breeder/cs/`, mirroring
  `cax.cs`. The top level is now `main` + `core/` + `cs/` + `report/` — the same
  split as the library, and adding a system no longer widens the top level.
- **Configs 60 → 15**: defaults = protocol (see PARAMETERS.md), settled one-offs
  deleted; regeneration verified byte-identical for every survivor except two
  called-out rows.

## Testing protocol

Each row is validated by an ablation pair (draft vs. draft-with-row-reverted) at matched
evaluation budget, ≥3 seeds, comparing: valid fraction, QD-style coverage proxy
(descriptor-space hypervolume / mean pairwise distance), best and mean fitness, and
phenotype variance. Update Tested/Improvement/Conf accordingly.
