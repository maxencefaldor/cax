# Changelog

All notable changes to CAX are documented here. Versions follow
[semantic versioning](https://semver.org): while CAX is pre-1.0, breaking changes
raise the minor version.

## 0.4.0

The first release since 0.3.3, covering both a backlog of unreleased work and a
library-wide review. It renames and removes public API — see **Migrating** below.

### Migrating from 0.3.3

| Before | After |
| --- | --- |
| `cs(state, num_steps=n, sow=True)` then `nnx.pop(cs, nnx.Intermediate)` | `cs(state, num_steps=n, return_states=True)` |
| `Lenia(spatial_dims, channel_size, ...)` (positional) | `Lenia(spatial_dims=..., channel_size=..., ...)` |
| `cax.cs.lenia.KernelParams`, `GrowthParams` | `LeniaKernelParams`, `LeniaGrowthParams` |
| `cax.cs.particle_lenia.KernelParams`, `GrowthParams` | `ParticleLeniaKernelParams`, `ParticleLeniaGrowthParams` |
| `BoidPolicy` | `BoidsPolicy` |
| `cs.render(state, boids_size=...)` | `cs.render(state, particle_radius=...)` |
| `ParticleLifeState.class_` | `ParticleLifeState.class_id` |
| `ParticleLife(..., A=...)` | `ParticleLife(..., attraction_matrix=...)` |
| `ParticleLenia` state as a bare array | `ParticleLeniaState(position=...)` |
| `Life(..., rngs=rngs)`, `Elementary`, `LangtonAnt`, `ReactionDiffusion` | drop `rngs`; these systems are deterministic |
| `Sandpile(padding="OPEN")` | `Sandpile(padding="ZERO")` |
| `cax.cs.lenia.patterns` pickles | `load_pattern(name)` |

Three changes alter numerical results without raising:

- `FlowLenia`'s `theta_A` now defaults to `channel_size`, reproducing the official
  implementation for multi-channel systems (it previously defaulted to `1.0`).
- `ParticleLife.render` now maps simulation-y-up to image-y-down, matching `Boids`;
  renders are flipped vertically relative to 0.3.3.
- `FlowLenia`'s Sobel gradients now wrap the periodic domain (see below).

### Added

- Flow Lenia works in any number of spatial dimensions; the 2D-only restriction is
  gone. Gradients use n-dimensional separable Sobel kernels and reintegration
  tracking generalizes to `(2 * dd + 1) ** num_spatial_dims` displacements.
- `cs(..., return_states=True)` returns `(final_state, states)`, the per-step states
  stacked as the scan's outputs.
- Boundary conditions are configurable on `Life`, `Elementary`, and
  `ReactionDiffusion` via `padding`, joining `Sandpile`.
- `cax.utils.dynamics` with `toroidal_difference` and `damped_euler_step`, shared by
  Boids and Particle Life.
- `cax.utils.numerics` with `safe_divide` and `safe_norm`, which keep gradients finite
  where an argument can degenerate.
- Rasterization helpers `pixel_grid`, `nearest_point`, and `soft_disk_mask` in
  `cax.utils.render`, shared by the three particle renders.
- `cax.cs` re-exports every system, and each system package documents its reference.
- `Pool.sample` takes `replace`, defaulting to sampling with replacement.
- `load_pattern(name)` and `PATTERN_NAMES` for Lenia's shipped patterns, stored as
  `.npz` archives instead of pickles.
- Leniabreeder example, and a VAE in `cax.nn`.

### Changed

- `FlowLenia` subclasses `Lenia`, inheriting its perception and rendering.
- Flow Lenia's Sobel gradients wrap the torus, consistent with its perception and
  reintegration. The official implementation zero-pads them, contradicting its own
  periodic domain; away from the boundary the two agree to float32 rounding. All
  Sobel gradients are computed by a single grouped convolution.
- Every constructor is keyword-only.
- Simulation states are frozen dataclasses registered as JAX pytrees, replaced with
  `dataclasses.replace` rather than mutated.
- Particle Lenia's default growth function is reference-faithful (`peak_growth_fn`);
  the previous default inflated the growth width by √2.
- Lenia kernel weights are normalized to sum to one, Chan's multi-kernel convention.
- `KernelParams.b` is `beta`.
- Perceive modules use an optimized pad-slice implementation, and Reaction-Diffusion's
  stencil is a plain constant rather than trainable weights.
- Python 3.12 or later is required.

### Fixed

- Boids no longer produces `nan` gradients for an empty neighborhood, and its
  steering goes through the safe numeric primitives.
- Langton's Ant stores position and direction as integers.
- `ParticleLenia.render` works again, and its `mode` is validated rather than
  silently falling back.
- Several notebook visualization cells discarded the returned states and rendered
  stale frames.

## 0.3.3 and earlier

See the [release history](https://github.com/maxencefaldor/cax/releases).
