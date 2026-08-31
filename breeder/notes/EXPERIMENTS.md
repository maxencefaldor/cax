# Experiments

## Ablation 2 — 2026-08-30, single-ancestor protocol, one run per GPU

**Protocol (new baseline)**: every run starts from ONE full soliton — the Aquarium
(`5N7KKM`) with its own catalogued rules and pattern (`init: soliton_full`,
`num_init = population_size = 1024`, all identical at start) — and mutates from there.
No random init, no fresh sampling during search (`sample_ratio 0`). Base geometry: 128², `state_scale 1`
(decided in ablation 1 + world_scale figure). `growth_std_range` widened to (0.005, 0.2)
to contain the Aquarium's catalogued values. Metrics in `R × state_scale` units;
per-generation `diversity` (reference pairwise), `vendi` (effective species count), and
`variance` logged. 1024 generations.

| GPU | Name | Change vs. base | Question |
|-----|------|-----------------|----------|
| 0 | base | — | reference (seed 0) |
| 1 | base_seed1 | seed 1 | seed variance |
| 2 | base_seed2 | seed 2 | seed variance |
| 3 | latent2 | latent_size 2 | best latent (D1) |
| 4 | latent4 | latent_size 4 | best latent |
| 5 | latent16 | latent_size 16 | best latent |
| 6 | latent32 | latent_size 32 | best latent |
| 7 | metrics | hand-crafted descriptor | learned vs manual descriptor |

Judged **diversity-first** (vendi + reference pairwise + reports), fitness second.
Earlier outputs deleted.

### Results (1024 generations, all 1024/1024 valid)

best / mean speed; diversity = reference pairwise; vendi = in-run values (pre-fix cosine
kernel — compressed toward 1; the regenerated reports' vendi chips use the corrected
correlation kernel):

| Experiment | Best | Mean | Diversity | Vendi (old kernel) | Variance |
|------------|------|------|-----------|--------------------|----------|
| base (seed 0) | 0.226 | 0.020 | 0.128 | 1.36 | 0.0043 |
| base_seed1 | 0.141 | 0.017 | 0.117 | 1.31 | 0.0045 |
| base_seed2 | 0.207 | 0.023 | 0.128 | 1.31 | 0.0047 |
| latent2 | 0.154 | 0.017 | 0.119 | 1.24 | 0.0024 |
| latent4 | 0.150 | 0.013 | 0.091 | 1.21 | 0.0028 |
| latent16 | 0.195 | 0.018 | 0.164 | 1.34 | 0.0056 |
| latent32 | 0.165 | 0.012 | **0.277** | 1.28 | 0.0111 |
| metrics | 0.221 | **0.084** | **0.785** | 1.24 | 0.0285 |

Corrected Vendi (correlation kernel, computed on the final populations by the reports):
base 7.4 / seed1 8.5 / seed2 7.4, latent2 6.3, latent4 7.8, latent16 6.0, latent32 4.8,
metrics 3.6.

Readings:
- **Seed noise dominates fitness**: base best spans 0.141–0.226 across three seeds. Any
  single-seed fitness gap below ~0.08 is noise; latent variants are fitness-equivalent.
- **Spread and species count trade off**: pairwise spread ranks 4 < 2 ≈ 8 < 16 < 32,
  but corrected Vendi ranks the other way — base (latent 8) holds ~7–8 effective
  species, latent32 only ~4.8, metrics ~3.6. Wide descriptor spaces push the population
  apart along fewer collective modes; latent 8 preserves the most distinct kinds.
  Tentative verdict: keep `latent_size 8` (pending wave-2 seeds and visual checks).
- **metrics descriptor is the outlier**: by far the largest spread and variance and the
  best mean fitness, but the fewest effective species — a smeared continuum, not a
  radiation. Judge in its report.
- Single-ancestor protocol keeps every individual valid in every run (the Aquarium
  lineage never degenerates).

### Wave 2 results (best / pairwise diversity / corrected vendi, all 1024/1024 valid)

- vgg_descriptor: 0.187 / 0.623 / **5.9** — the pretrained perceptual descriptor gives
  metrics-like spread with nearly twice metrics' species count at competitive fitness.
  Best spread×species×fitness combination so far.
- deepdream_c7 / c42 / c141 (fitness = VGG16 activation, own units): vendi 6.9 / **8.9**
  / 7.9 — c42 is the most speciated population of the whole campaign.
- Seeds: base now 5 seeds (vendi 7.4/8.5/7.4/8.2/8.8, mean ≈ 8.0); latent16 3 seeds
  (6.0/4.7/6.9); latent32 2 seeds (4.8/5.0); metrics 2 seeds (3.6/3.0, best fitness
  0.22/0.25); latent2 2 seeds (6.3/7.1); **latent4 2 seeds (7.8/10.75, mean ≈ 9.3)** —
  latent4_seed1's 10.75 is the campaign's highest, reopening latent 4 vs 8.

### Wave 3 results — the settled picture (corrected vendi, mean ± spread over seeds)

| Config | Vendi over seeds | Mean | Best fitness range |
|--------|------------------|------|--------------------|
| latent2 (n=3) | 6.3, 7.1, 4.4 | 5.9 | 0.15–0.19 |
| latent4 (n=5) | 7.8, 10.8, 8.4, 2.6, 7.5 | 7.4 (σ≈3.0) | 0.11–0.17 |
| **latent8 (n=5)** | 7.4, 8.5, 7.4, 8.2, 8.8 | **8.0 (σ≈0.6)** | 0.13–0.23 |
| latent16 (n=4) | 6.0, 4.7, 6.9, 7.6 | 6.3 | 0.14–0.20 |
| latent32 (n=2) | 4.8, 5.0 | 4.9 | 0.14–0.17 |
| metrics (n=3) | 3.6, 3.0, 3.7 | 3.4 | **0.22–0.27** |
| vgg_descriptor (n=2) | 5.9, 5.7 | 5.8 (spread ≈0.6) | 0.19–0.21 |
| deepdream_c42 (n=2) | 8.9, 4.6 | 6.8 | activation units |

**Verdicts:**
- **Latent 8 is the answer** (ledger D1 closed): best mean speciation AND by far the
  most reliable (σ 0.6 vs latent4's σ 3.0 — latent4 can hit the campaign record 10.8 or
  collapse to 2.6). Paper's choice confirmed under the modern stack.
- **metrics descriptor**: reproducibly the fitness champion (0.22–0.27) and the
  speciation floor (~3.4). A quality-vs-diversity dial, not a default.
- **vgg_descriptor reproduces well**: ~5.8 species, metrics-like spread, competitive
  fitness in both seeds — the strongest quality×spread×species compromise. Candidate for
  a future default descriptor, pending visual judgment of its reports.
- deepdream speciation is seed-dependent (8.9 vs 4.6); the galleries remain the point.


## Ablation 1 (archived) — 2026-08-30, one run per GPU, seed 0

Launched via `breeder/scripts/ablation_1.sh`. Baseline: defaults of `breeder/main.py`
(64² world, state_scale 2, latent 8, noise seeding, pure mutation, fitness = speed,
population 1024, 256 child individuals × 256 generations, 8× oversampled init).

| GPU | Name | Change vs. base | Ledger row |
|-----|------|-----------------|------------|
| 0 | base | — | reference |
| 1 | world128_s1 | 128² world, state_scale 1 (official world/kernel ratio ≈ 10.7) | S2 |
| 2 | world128_s2 | 128² world, state_scale 2 (ratio ≈ 5.3) | S2 |
| 3 | latent2 | VAE latent size 2 (draft value) | D1 |
| 4 | metrics | Hand-crafted descriptor (mass, speed), no VAE | 6.x / sanity |
| 5 | soliton | Seeding from {5N7KKM, VT049W} solitons | G4 |
| 6 | immigrants05 | sample_ratio 0.5 (half fresh samples per batch) | emitter mix |
| 7 | random_search | sample_ratio 1.0 (pure random search, no mutation) | emitter mix |

Notes:
- 128² experiments use one extra VAE conv layer (features 3,16,32,32) to keep the flattened
  encoder width at 8192, same as the 64² experiments.
- The metrics experiment normalizes mass by 10 and speed by 0.01 so the two descriptor axes are
  commensurate for DNS distances; scales are rough, revisit if this experiment setup is kept.
- Comparison criteria (see MODIFICATIONS.md testing protocol): valid fraction, descriptor
  spread, best/mean fitness, and — decisive — visual quality of discovered creatures.

### First launch (256 generations, pre-refactor code) — interim observations

The two 128² experiments OOM'd (unchunked population re-encoding); the six others completed in
~3 minutes each (28 s for the VAE-free metrics experiment). Final (best speed, mean speed,
valid/1024):

| Experiment | Best | Mean | Valid |
|-----|------|------|-------|
| base | 0.108 | 0.0032 | 1024 |
| immigrants05 | 0.111 | 0.0025 | 1024 |
| latent2 | 0.073 | 0.0067 | 1024 |
| metrics | 0.100 | 0.0309 | 1024 |
| soliton | 0.075 | 0.0018 | 1024 |
| random_search | 0.029 | 0.0015 | 436 |

Early signals: latent 8 > latent 2 on best fitness; fresh sampling ≈ base; pure random search
far behind on fitness (as expected — its value would be diversity, judge visually);
metrics descriptor concentrates the population at high fitness (mean 10× higher) but
likely with much lower visual diversity. All superseded by the relaunch below.

### Relaunch (1024 generations, refactored code, orbax checkpoints)

Same 8 experiments via the updated `ablation_1.sh` (pydantic config syntax), 1024 generations,
128² experiments fixed by minibatched re-encoding and init downselection.

**Status: temporary, pipeline-validation results.** Decisions here are fitness-only and
single-seed; the user's primary criterion is diversity. All experiments are rerun once
the diversity evaluation is trusted (reference space with fixed-physical-window crops,
metrics unit fix), and no architecture decision is final until judged on diversity.
Also note: `world128_s2` numbers below predate the metrics unit fix (velocities inflated
2× at state_scale 2 relative to state_scale 1).

Final results, seed 0 (best speed / mean speed, all 1024/1024 valid):

| Experiment | Best | Mean | Wall time |
|------------|------|------|-----------|
| base | 0.186 | 0.0023 | 15 min |
| world128_s1 | **0.233** | 0.0184 | 34 min |
| world128_s2 | 0.204 | **0.0354** | 34 min |
| metrics | 0.158 | 0.0588 | 1.5 min |
| latent2 | 0.102 | 0.0045 | 15 min |
| immigrants05 | 0.080 | 0.0030 | 15 min |
| soliton | 0.073 | 0.0020 | 15 min |
| random_search | 0.041 | 0.0017 | 15 min |

Readings (single seed, fitness only — diversity to be judged with the new reference
metrics and reports):
- **World size (S2)**: both 128² worlds dominate the 64² base on best *and* mean fitness
  (mean 8–15×) — the 64² world is too small relative to the kernel; see
  `notes/figures/world_scale.png`.
- **Latent size (D1)**: latent 8 (0.186) > latent 2 (0.102).
- **Seeding (G4)**: noise (0.186) > soliton lineages (0.073) on fitness.
- **sample_ratio**: neutral at 256 generations but clearly harmful to best fitness by
  1024 (0.080 at 0.5, 0.041 at 1.0) — fresh sampling dilutes exploitation on long horizons.

## Descriptor geometry drives mass (2026-08-30, net-velocity wave)

Measured: vgg_descriptor population median mass 18.4 (max 38.7) vs metrics 11.6 —
both far above the ancestor Aquarium (~2.4). Mechanism (confirmed by user's visual
read): validity has no upper mass bound (min_mass is a death gate only), fitness
charges nothing for size, so the descriptor decides. The metrics descriptor holds
light-creature niches open (mass is an explicit axis); the 256-D VGG space has no
mass axis and rewards perceptual distinctness, for which growth+texture is a cheap
route — the population inflates until the concentration gate is the only ceiling.
Consequences: S2 (torus self-interaction) is back in play for the heaviest vgg
individuals; candidate counter-lever without new gates: a mass axis alongside vgg,
`series: [[vgg, 1.0], [mass, 10.0]]`.

## Micro-dot diagnosis resolved: the premise was wrong (2026-08-31)

The flag ("base AURORA run is shit — micro-dots, something broke in the VAE path")
triggered a 5-arm factorial plus git archaeology of every commit between the ablation-2
wave (14:18) and the flagged wave (21:59). Verdict: **nothing in the VAE path changed or
broke** — configs byte-identical on every VAE field, `cax.nn.vae` and `cax.cs.lenia`
untouched, the VAEDescriptor→Encoder refactor verified semantically identical.

The premise didn't survive the data: VAE-latent runs have **always** sat at median mass
~1.0–4.0, including the "good" ablation-2 runs (base 14:18: mass med **1.60**, Vendi 7.4).
The flagged base (21:59) is actually *heavier* (2.95). Mass never discriminated good from
bad — the gap vs metrics (11.6) is descriptor geometry (see previous section), present in
both eras. Factorial arms (mass med / best): steps1k 1.6/0.071, narrow_r 3.4/0.061,
narrow_growth 1.85/0.054, homeostasis_v2 1.8/-0.0, inst_fitness 1.5/**0.2376**,
steps16k pending.

What actually changed between the waves (4 semantic deltas): kernel_r_range widened,
growth_mean_range widened, num_init 1024→1 (no-op: deterministic single-pattern
soliton_full made ablation-2's 1024 inits identical clones), and **the fitness
redefinition** `mean linear_velocity` → `norm_mean velocity`. Only the inst_fitness arm
reproduces ablation-2's numbers (best 0.2376 vs 0.2259) — the 4× "fitness collapse" is a
*definition* change (net displacement ≤ path length), not a regression, and best-fitness
values are **not comparable across fitness definitions**. Remaining honest question:
whether the *visual* judgment "base is shit" tracks the norm_mean fitness shifting the
population toward straight-line gliders, or expectation recalibrated by the
metrics/vgg breakthroughs — to be settled by eye against the ablation-2 base report.
Sanity rerun of metrics under current code reproduced the breakthrough regime
(mass med 10.63, best 0.1029): shared pipeline healthy.

## Two new complex systems (2026-08-31, overnight)

**Flow Lenia** and **Particle Life** integrated behind the same `ComplexSystem` carrier;
`core/motion.py` now supplies the motion series to all three systems, so a fitness
written against `velocity` transfers unchanged.

*Particle Life* needed a new observation vocabulary, since particles never die and mass
says nothing. Four independent axes — `concentration` (localization), `clustering`
(local density against the uniform expectation, 1 = soup), `radius`, `polarization`
(velocity alignment) — plus `thermal_velocity`. Validated against constructed
configurations:

| configuration | concentration | clustering | polarization |
|---|---|---|---|
| uniform soup | 0.001 | 0.99 | 0.00 |
| tight soliton | 0.985 | 14.15 | — |
| loose blob | 0.686 | 6.29 | — |
| **two distant clusters** | **0.000** | **7.06** | — |
| soliton marching / soup milling | — | — | 1.00 / 0.04 |

The two-cluster row is why `clustering` exists: concentration alone cannot tell it from
a soup. Requiring both gates rejects it, which is correct when the target is *one*
soliton.

Calibration (512 genotypes, settled at 1024 steps): concentration p50 0.36, clustering
p50 4.0; gates 0.5 / 2.0 keep 31%. Seeding measured at **55% viable from a blob vs 0%
from a uniform scatter** — a soup essentially never self-organizes into a localized
soliton within budget, so the localized seed is what makes the search possible.
Relaxation completes by ~1024 steps; at 256 the search would select on the transient.

*Flow Lenia* caveat, seen immediately in its first run: mass is conserved and the seed
is localized, so **1023/1024 individuals are valid** and net velocity sits at ~0.002 —
the gate does no work and the fitness landscape is nearly flat. Its degeneracies are not
Lenia's (nothing dies, nothing disperses), so both the validity notion and the objective
likely need rethinking for it — an open question, not a settled setup.

## Roll vs no-roll (2026-08-31): full-torus roll wins clearly

Tuned VAE (lr 1e-3, grad clip 1.0, 8192 steps), D4-invariant encoding, velocity fitness,
latent descriptor; the only difference is whether training augmentation includes the
full-torus roll.

| | best | vendi | mass med | mass p90 | visual |
|---|---|---|---|---|---|
| `d4 + roll` | **0.0750** | 4.76 | 1.86 | **4.24** | varied: rings, arcs, elongated forms, multi-part creatures |
| `d4` only | 0.0649 | 6.09 | 1.80 | 3.32 | **monotonous**: nearly every individual the same small concentric ring |

Seed 1 agrees on fitness (0.0918 vs 0.0507). The OOD worry about full-range roll —
frames are centered at encode time, so wrapped training views are off-distribution — is
real for *reconstruction* (roll costs ~12% validation loss on canonical frames) but the
search outcome goes the other way: the roll-trained descriptor holds a far more varied
population, while the no-roll one collapsed onto a single motif. Reconstruction quality
and descriptor quality are not the same objective.

**Vendi disagrees with the eye for the third time** (6.09 for the monotonous population
vs 4.76 for the varied one), after ranking the "shit" base run highest and the
breakthrough metrics run nearly lowest. Reference-space Vendi should not be used to
rank runs on diversity.

**Confirmed by eye (2026-08-31).** Reviewing the four reports directly, the verdict is
`roll` > `no-roll` — the same direction as fitness and the opposite of Vendi. Full-torus
roll is settled as the default augmentation; it is not revisited without a reason.

### Particle Life first results (2026-08-31)

Both arms searched from a single blob seed, 1024 generations, metric descriptors.

| arm | fitness | best | population medians |
|---|---|---|---|
| `particle_life_metrics` | `norm_mean velocity` | **9.63** | clustering 8.94, concentration 0.69, polarization 0.85, radius 1.01 |
| `particle_life_polarized` | `mean polarization` | **1.000** (saturated) | clustering 9.10, concentration 0.67, polarization 1.00, radius 1.01 |

Both populations are genuinely *solitons*, not soups: median clustering ~9 (a uniform
scatter is 1) and concentration ~0.68 at radius ~1 interaction radius. The two objectives
produce visibly different morphologies — the velocity arm yields comet-like streams and
sparse arcs with trailing particles, the polarization arm compact layered discs with
banded classes, which is what coherent drift of a mixed-class cluster looks like.

Caveat: polarization saturates at its maximum 1.0, so after that the objective stops
differentiating and diversity alone drives the run — the same shape of problem as the
homeostasis fitness. It is probably better used as a descriptor axis than as an
objective.
