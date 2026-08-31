# Experiments — what was measured, and what was judged

Only results that cannot be reconstructed live here. Two kinds qualify:

- **Judged by eye.** The user's visual verdict on a population. The runs behind these
  were deleted on 2026-08-31; nothing can recover the judgment but this file.
- **Measured at cost.** Numbers that took GPU-hours to produce.

Everything else — parameter values, code structure, how a conclusion was reached — is
either in the code or not worth keeping. Superseded waves are in git history.

## Judged by eye (the oracle)

**Pose invariance is the lever that made the search work** (2026-08-31). Training
augmentation over the dynamics' own symmetry group (`d4` + full-torus `roll`) plus
D4-orbit-averaged encoding. User's verdict, unprompted: *"the data augmentation tricks
made a massive difference."* Before, base VAE-latent runs produced incoherent micro-dot
populations; after, coherent creatures. Both levers are defaults.

**Full-torus roll beats no-roll** (2026-08-31, 2 seeds each, tuned VAE otherwise
identical). Judged directly from the reports: the roll arm holds rings, arcs, elongated
and multi-part creatures; the no-roll arm collapsed to nearly every individual being the
same small concentric ring. Fitness agrees (0.0750/0.0918 vs 0.0649/0.0507). The OOD
worry was real for *reconstruction* — roll costs ~12% validation loss on canonical
frames — but reconstruction quality and descriptor quality point in opposite directions.

**Concentration gate 0.5, not 0.1** (2026-08-30). Judged via a blur ladder, a gate-0.05
QD map, and a gate-0.1 A/B: 0.5 right, 0.4 acceptable, 0.1 too low. The +68% Vendi at
gate 0.1 was therefore counting non-life.

**The Particle Life prior beat both evolved populations** (2026-08-31). The random
sample sheet looked more varied to the user than either evolved arm. The 2-axis metric
descriptor appears to collapse visual variety. Unresolved — see open questions.

## Measured at cost

**Vendi disagrees with the eye, four times** (2026-08-30/31). It ranked the "shit" base
run highest, the breakthrough metrics run nearly lowest, the monotonous no-roll
population (6.09) above the varied roll one (4.76), and non-life above life at gate 0.1.
This is why the reference space was rebuilt (below) and why **no diversity metric
outranks the user's report review**.

**The reference space must be perceptual, not random** (2026-08-31). All four
roll/no-roll populations rescored in both spaces against the eye's verdict:

| measure | seed 0 | seed 1 |
|---|---|---|
| random CNN, pairwise distance | ✓ | ✗ |
| random CNN, Vendi (cosine kernel) | ✗ | ✓ |
| random CNN, Vendi (covariance kernel) | ✓ | ✓ (margin 2.5%) |
| **VGG16 layer 11, any of the three** | ✓ | ✓ (margin 1.6–2.1×) |

Every VGG measure agrees on both seeds; every random-space measure fails one. A random
projection preserves *pixel* distance, and near-identical creatures differing in phase
and micro-detail are far apart in pixel space. Two independent fixes, both shipped: VGG
features, and a covariance kernel for Vendi (unit-rescaling discards how far an
individual sits from the mean, so a collapsed population's residual noise —
near-orthogonal in high dimension — scored near maximum). The old cosine kernel stays
logged as `vendi_cos` so future disagreements stay visible.

**Mutation operators inherited from the draft are pathological** (2026-08-31, simulated
over 256 lineages):

- *Dirichlet weight drift fixates.* `Dir(100·w)` is mean-preserving, but a bounded
  martingale on the simplex converges to its corners, and a weight near 0 gets alpha
  near 0 — absorbing. 6 of 8 rules dead (<1e-6) by ~100 applications, all but one by
  ~300 — well inside one 1024-generation run. A Jeffreys-style pseudo-count
  `Dir(100·w + 0.5)` leaves **zero** dead rules after 1000 applications.
- *The concentration was never sized.* At `c=100` a weight takes 27% relative steps per
  mutation, the largest in the genotype, while every other parameter takes ~1% of its
  range; `c=1600` restores parity.
- *The notebook's beta normalization is a bug.* It takes `nanmax` across **all** kernels,
  so after one mutation only the globally-tallest kernel keeps peak 1.0 — breaking the
  invariant its own sampler establishes. breeder normalizes per kernel.
- *Seed mutation creates mass.* One-sided reflection at 0 makes every empty pixel gain
  `sigma*sqrt(2/pi) ≈ 0.008` per mutation in expectation. The multiplicative operator
  (`x * exp(sigma*noise)`) is support-preserving: empty stays exactly empty.

**Sampling priors: width-first, measured** (2026-08-30, 2048 noise-init samples per
variant). Tightening `kernel_r_range` monotonically *reduced* viability (3.3% at low
0.2, 1.9% at 0.5), refuting the analytical suspicion that sub-4px kernels are wasted;
widened to (0.1, 1.2), viability held at 3.4% with slightly more species. Widening
`growth_mean_range` upper 0.5 → 0.7 gave **60% more life** (3.3% → 5.3%) with species
held — the catalogued envelope's 0.011 margin was hiding life above 0.5.

**Particle Life calibration** (2026-08-31, 512 genotypes). Relaxation completes by ~1024
steps (concentration 0.90 → 0.40, clustering 12.7 → 4.0); at 256 steps the search would
select on the transient. Gates 0.5 / 2.0 keep 31%. Seeding matters absolutely: **55%
viable from a blob vs 0% from a uniform scatter** — a soup essentially never
self-organizes into a localized soliton within budget.

**VAE training was systematically under-trained** (2026-08-31, 24-config grid on 2048
curated creatures plus a live-run probe). Settled: lr 1e-3 constant, grad clip 1.0,
8192 steps. Going to 16384 steps buys ~3.4% more reconstruction — not visible in the
search.

**Homeostasis and polarization objectives saturate.** Homeostasis reached exactly
0.0000 on both seeds; Particle Life polarization reached 1.0. Once saturated the
objective stops differentiating and diversity alone drives the run. Both are better
used as descriptor axes than as objectives.

**Descriptor geometry drives body mass** (2026-08-30). `vgg_descriptor` median mass 18.4
vs `metrics` 11.6, against an ancestor Aquarium at ~2.4. Validity has no upper mass
bound and fitness charges nothing for size, so the descriptor decides: the metrics
descriptor holds light-creature niches open because mass is an explicit axis, while the
256-D VGG space has no mass axis and rewards perceptual distinctness, for which growth
and texture are a cheap route.

**A fitness redefinition is not a regression** (2026-08-31). A suspected 4× "fitness
collapse" was traced to `mean linear_velocity` → `norm_mean velocity` (net displacement
≤ path length). Best-fitness values are **not comparable across fitness definitions**.
The related micro-dot premise also failed: VAE-latent runs always sat at median mass
1.0–4.0, including the runs judged good.

## Open questions

1. **The mutation wave.** Four arms (`lenia_weight_floor`, `lenia_weight_tuned`,
   `lenia_state_frozen`, `lenia_state_multiplicative`) plus seed replicates, running
   2026-08-31. At generation 192 no arm is separated from base — base's own two seeds
   differ 40% on variance while agreeing to 0.5% on fitness, so seed spread is
   comparable to every effect. `state_frozen` is the one to watch: if mutating only ~30
   rule parameters matches mutating 49,000, seed evolution is dead weight.
2. **Flow Lenia has no working notion of degeneracy.** 1023/1024 valid, best fitness
   0.0019 — mass is conserved and the seed is localized, so the validity gate does no
   work and the landscape is nearly flat. Both the gate and the objective need
   rethinking for it.
3. **Particle Life: the prior beat evolution** to the user's eye. `particle_life_sampled`
   (50% fresh samples) is one counter-arm; the learned descriptor is another. Unresolved.
4. **Latent size is unsettled, and its old verdict is suspect.** "Latent 8" (5 seeds,
   Vendi 8.0 ± 0.6 vs latent4's 7.4 ± 3.0) was decided on Vendi in the random reference
   space — the instrument later shown to disagree with the eye four times. A fresh
   comparison at 4/8/16 gave best 0.0666 / 0.0750 / 0.0603: no trend. Treat as open.
5. **Retrain-from-scratch vs continuous VAE training** — never tested against each other.
6. **The sub-4px kernel tail.** `kernel_r_range` lower bound 0.1 gives 1.2px kernels at
   `state_scale 1`, grid-unresolvable. Measured harmless, never revisited with QD
   evidence.
