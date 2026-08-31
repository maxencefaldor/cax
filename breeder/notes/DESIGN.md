# Design — breeder

Quality-diversity search over CAX complex systems.

## Invariants (agreed, frozen)

- **Lenia fidelity**: standard multi-channel Lenia — Gaussian ring kernel core (std 0.15)
  and exponential growth `2·bell(u, m, s) − 1`, exactly as the official code. Kernel and
  growth function shapes are never searched.
- **R, T, state_scale**: fixed, never evolved — discretization/scale parameters,
  phenotype-neutral in the continuum limit (search noise).
- **channel_size, kernel count and topology**: fixed within a search — varying them
  changes the universe, not the creature; results would not be comparable.
- **CAX vocabulary**: the simulated system is a *complex system* (never "substrate");
  `breeder/lenia` instantiates CAX's `Lenia` complex system directly inside `develop`.
- **Flax NNX only**, never linen. Mesh + sharding constraint + jit for multi-GPU.
- **One experiment = one config**: a pydantic `Config` fully specifies a run and is
  stored as `config.yaml` in the run directory (YAML for configs everywhere). No pickle
  anywhere; checkpoints are asynchronous orbax (written, never read back — see
  Implementation notes). The `encoder` section and the complex system are
  **discriminated unions** on `name` (`extra="forbid"` everywhere — a field on the
  wrong variant is a validation error, never silently ignored); fitness and descriptor
  are plain reduction configs. Configs from earlier schemas are not readable —
  regenerate with `scripts/make_configs.py`.

## Vocabulary

Lean, classical GA + CAX conventions. No operator objects, no invented nouns.

| Term | Meaning |
|------|---------|
| `Genotype` | Structured pytree: rule params + `state_init`. |
| `sample(key, config)` | Fresh genotype from the prior. |
| `mutate(key, genotype, config)` | Variation. (`crossover` slots in later if wanted.) |
| `develop(genotype, config)` | Genotype → phenotype: instantiate the CAX complex system from the rule params, simulate, observe. |
| `Phenotype` | Observable outcome of development: `frames` (uint8 video, one per step) + `series` (named time series) — the complete record. During evaluation the experiment's encoder enriches `series` with its learned feature series. The only thing fitness/descriptor ever see — this keeps the search core independent of the complex system. |
| `valid(phenotype, config)` | False if the phenotype degenerated. |
| `config.sample.strategy` | Genotype sampling strategy, a registry of `(key, config) -> Genotype` functions: `noise` (random rules + noise blob), `soliton` (random rules + known pattern), `soliton_full` (a soliton's own rules + pattern — official lineage breeding), later `learned`. A complex system samples or mutates — "init" is the QD algorithm's concept (`dns.init` seeds the population *using* `sample`), so the word never appears in a complex system. `sample:` and `mutate:` are symmetric sub-configs; parameter ranges stay top-level because they define the genotype space both operators share. |
| `sample_ratio` | Fraction of child individuals drawn fresh from `sample` (`sampled_individuals`); the rest are `mutated_individuals`. 0.0 = pure mutation, 1.0 = pure random search. A float, not an abstraction. |
| `series` | A named time series of the phenotype: emitted by `develop` (mass, linear_velocity, ...) or by a named encoder (latent, vgg, ...). The one noun fitness and descriptor consume; references are validated up front. |
| `Encoder` | `nnx.Module`, `encode(frames) -> {series_name: (window, feature_size)}`; the experiment's only weight carrier, declared as the single `encoder:` in the config (`null` = the no-op base `Encoder`, emitting nothing — encoder-less runs share the code path). Each type emits a fixed series: `vae` → `latent` (trainable via `fit`, refitted from scratch every `train_interval` — AURORA), `vgg` → `vgg` (fixed pretrained). One encoder always suffices since deepdream (vae descriptor + vgg fitness) was retired (2026-08-31). |
| `Fitness` | Weightless reduction of one named series to a scalar, **maximized**. Specified either literally — `{series, reduce (mean|var|norm_mean), sign, channel?, window}`, the formula in the code's own series vocabulary (net travel = `norm_mean velocity`; a VGG channel = `mean vgg[c]`) — or by a **named objective** for definitions that are paper-pinned or composite (`{name: homeostasis, series: latent}`, arXiv:2406.04235, battery-asserted against the hand-derived formula). Names are earned by citation or non-trivial semantics, never invented as aliases for transparent reductions. |
| `Descriptor` | Weightless reduction of named series to a vector: each `(series, scale)` pair contributes its windowed time-mean divided by `scale` (axes commensurate). AURORA = `[[latent, 1.0]]`; hand-crafted = `[[mass, 10.0], [linear_velocity, 0.01]]`. Re-encoded from archived observations after an encoder refit. |
| `window` | Number of final steps a fitness, descriptor, or encoder consumes, excluding the developmental transient. Same name, same meaning, everywhere. The archive (`observations`) spans the encoder window. |
| `minibatch_size` | Individuals processed per sequential minibatch in evaluation/encoding (bounds peak memory; the count of minibatches is derived at runtime). |
| `ask` / `tell` | Algorithm interface (DNS first; MAP-Elites later). `ask` = selection + variation. |
| `num_children` | Child individuals per generation (GA's lambda; `population_size` is mu). One `individual`; `child_individuals`, `parent_individuals`, `sampled_individuals`, `mutated_individuals` when the distinction matters. |
| Layout mirrors `cax` | `breeder/{core, cs}` mirrors `cax/{core, cs}`: `core/` is the system-agnostic search, `cs/` holds one package per complex system. A system package never imports `core`'s search, and `core` never imports a system. |
| Lenia family | Flow Lenia is Lenia with a mass-conserving update: `FlowLeniaConfig` subclasses `LeniaConfig`, and `flow_lenia` re-exports Lenia's genotype half (`Genotype`, `sample`, `mutate`, `valid`) plus the shared `observe` — all through `lenia`'s public API. Editing Lenia's genotype space or observation edits Flow Lenia's, by design. |
| `*_fn` vs bare name | A module or callable takes the `_fn` suffix (`fitness_fn`, `descriptor_fn`, `encoder_fn`, `reference_fn`, `sample_fn`, `mutate_fn`); the bare name is the value it produces (`fitness`, `descriptor`). Config sections keep bare names (`config.fitness` is a `FitnessConfig`). |

Each complex system is a module of these functions (`breeder/lenia`: `sample`, `mutate`,
`develop`, `valid`), used as a namespace: `lenia.sample(key, config)`. The system's
config (`LeniaConfig`, tagged `name: lenia`) binds them in `build()` and returns
`core.ComplexSystem` — the four functions plus `spatial_dims` (frame geometry), `unit`
(physical unit in px, Lenia's `R·state_scale`, for reference crops), and `series` (the
time-series names `develop` emits, validated against fitness/descriptor references up
front), plus `num_frames` (how many final frames it renders). `main.py` and `report.py`
touch only this carrier; adding a system means adding one module and one line in the
`complex_system` union — the core never imports a system.

Three systems are integrated. **Lenia** is the reference. **Flow Lenia**
(arXiv:2212.07906) is Lenia with a mass-conserving update, so its genotype space is
*identical* and its package reuses Lenia's `Genotype`, `sample`, `mutate` and `valid`,
adding only a config (subclassing `LeniaConfig`) and a `develop`. **Particle Life**
brings its own genotype — attraction matrix, force crossover `beta`, initial
arrangement — and its own observation problem: particles never die, so mass says
nothing and *organization* has to be measured instead (see `particle_life/develop.py`).
Motion series are shared through `core/motion.py`, so `velocity` means the same thing
in all three and a fitness transfers between them unchanged.

## Layout

```
breeder/
  core/         # Phenotype, Encoder, Fitness, Descriptor, DNS, motion, variation, checkpointing
  cs/           # the complex systems, mirroring cax.cs
    lenia/        # config, genotype, sample, mutate, develop/valid
    flow_lenia/   # config, develop  (genotype space is Lenia's, reused wholesale)
  particle_life/# config, genotype, sample, mutate, develop/valid
  main.py       # composition root: system registry, Config/CliConfig, the search loop
  scripts/  configs/  notes/  output/
```

**Configs live with what they build**: each class sits beside its config in its
`core/` module (`VAEEncoder` + `VAEEncoderConfig`, ...), and each complex system's
config lives in its own package — so new variants never grow `main.py`, which keeps
only the system registry (`ComplexSystemConfig`), the experiment `Config`/`CliConfig`,
and the loop.

## Implementation notes

- Config via pydantic; the command line via pydantic-settings on a separate adapter
  `CliConfig(Config, BaseSettings)` (nested flags: `--complex-system.sample.strategy soliton`,
  tuples as `"[128,128]"`, the encoder section as JSON). The split is load-bearing:
  validating a `BaseSettings` model re-applies CLI sources over the data, so a pure
  `Config` is the only safe target for `load_config` — files stay exactly what they
  say. Checkpoints (DNS state, encoder params, rng key) are written every
  `checkpoint_interval` but never read back: runs are cheap, so resume code was removed
  (2026-08-31, user-directed) — the artifacts remain, so resuming could be rebuilt.
- The encoder architecture: the single encoder is the experiment's weight carrier and
  enriches each phenotype's `series`; fitness and descriptor are symmetric weightless
  reductions over named series, validated against `cs.series` + the encoder's series.
  The AURORA loop retrains the encoder (`fit`) and re-encodes the population's
  descriptors from the archived observations when the descriptor reads the trainable
  series.
- Evaluation is sharded over the child individuals axis on a mesh of all visible devices
  (`with_sharding_constraint` inside the jitted evaluate); modules are replicated. Grid
  searches pin one run per GPU via `CUDA_VISIBLE_DEVICES`; a single big run scales by
  raising `num_children` (must divide the device count) across the whole mesh.
- The oversampled init (`num_init`) bootstraps the VAE, then is immediately
  downselected to `population_size` under dominated novelty — bounds memory and keeps one
  compiled shape.
- VAE training samples random (individual, frame) batches for a fixed number of steps
  (official scheme). Evaluation and population re-encoding process `minibatch_size` individuals per sequential minibatch via `jax.lax.map(..., batch_size=minibatch_size)`
  (encoding everything at once OOM'd at 128²).
- Degenerate-but-encodable phenotypes still get descriptors; validity only gates fitness.
- `train_interval` is a top-level experiment field: the refit cadence of the trainable
  encoder and the length of the compiled scan block in the main loop. Per-generation
  metrics leave the device through a `jax.debug.callback` inside the scan (measured
  free: 510s vs 505s on the metrics run, bitwise-identical trajectory) — `log.csv` rows
  carry true per-generation wall time and the console streams live.

## Diversity evaluation

No single number is "diversity"; we log a complementary battery, all computed in the
fixed reference space (frozen random CNN over centered, fixed-physical-window crops —
comparable across generations, experiments, and world sizes):

- `diversity` — mean pairwise distance: overall spread; blind to clustering (two tight
  far-apart clusters score high).
- `vendi` — Vendi score (Friedman & Dieng 2023): exponential entropy of the similarity
  kernel's eigenvalues = *effective number of distinct species*. The most interpretable
  and increasingly standard diversity measure in ML; punishes duplicates, so it
  complements pairwise distance.
- `variance` — pixel-space population variance, the official Leniabreeder measure
  (continuity with the paper).
- The reports add the human check: thumbnails + the map. Subjective alignment is the
  open target — a pretrained perceptual space (the `vgg` features below) is the natural
  next reference to correlate with human judgment.


## Open questions

- Logging: adopt the MetricWriter pattern (clu; TensorBoard + JSONL sinks, logging for
  events) — assessed and postponed 2026-08-30 (~+20 lines, buys cross-run dashboards).

- Deepdream-style fitness: maximize a chosen pretrained-network feature as the fitness
  (fun, not priority) — trivially expressible now as a `Fitness` over `vgg` features.

- DNS descriptor re-encoding cadence vs. VAE training schedule (ledger D2).
- World size / state_scale fix for the world-to-kernel ratio (ledger S2) — ablation 1.
- Dirichlet mutation with near-zero weights can produce NaN samples (alpha → 0); such
  child individuals are discarded as invalid. Consider an alpha floor.
