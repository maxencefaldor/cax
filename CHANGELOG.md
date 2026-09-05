# Changelog

All notable changes to CAX are documented here.
Versions follow [semantic versioning](https://semver.org): while CAX is pre-1.0, breaking changes raise the minor version.

## Unreleased

### Changed

- RGBA arrays are premultiplied: colour is scaled by alpha, so a pixel holds the light it emits and a transparent pixel is zero in every channel.
  `get_emoji_array` returns targets in this form and `rgba_to_rgb` composites them over white as `rgb + (1 - alpha)`; `render_array_with_channels_to_rgba` premultiplies the colour it builds.
  The Growing NCA notebooks trained against straight-alpha targets, whose anti-aliased edge pixels store full-strength colour behind almost no alpha --- colour an alive-masked automaton cannot reproduce, and which a mean squared error on the RGBA channels weighs the same as colour that is seen.
  Three notebooks premultiplied by hand and five did not; now the loader does it, as the reference implementations do.
- `examples/62_growing_nca_es.ipynb` trains on the Growing NCA setup: a 72x72 grid, 128 steps, the official seed, and a loss on the second half of the rollout rather than three checkpoints.
  A loss on the final frame alone leaves the growth unconstrained and evolution settles for a blob; scoring every frame of the second half grows the gecko.
  The strategy is Open-ES with a population of 512 and a constant standard deviation, and the update takes half steps: at full steps the seed dies for every member and the fitnesses tie.
- `examples/43_diffusing_nca.ipynb`, `examples/49_isotropic_nca.ipynb` and `examples/63_growing_nca_rl.ipynb` play the trained automaton for twice the training horizon, as the other Growing NCA notebooks do.
- `examples/32_neural_particle_automata.ipynb` shows its premultiplied target composited over white.

## 0.4.3

### Added

- `examples/49_isotropic_nca.ipynb` --- a neural cellular automaton with no preferred direction, after [Growing Isotropic Neural Cellular Automata](https://arxiv.org/abs/2205.01681).
  Every cell carries its own orientation and reads gradients rotated into that frame, so the pattern is grown at an angle nobody chose; the loss compares in polar coordinates and minimises over every rotation.
- `examples/50_difflogic_ca.ipynb` --- a cellular automaton whose rule is made of logic gates, after [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/).
  Each gate is a softmax over the sixteen boolean functions of two inputs while training and an `argmax` afterwards, which leaves a discrete circuit.
  It recovers Conway's Game of Life exactly: all 512 neighbourhoods, and every cell of a 64x64 board over 64 steps.
- `grad2_kernel` takes a `neighborhood`, `"von_neumann"` or `"moore"`.
  The default stencil favors the grid axes, which a system asked to behave the same in every direction picks up on; the Moore diagonals cut that bias from 50% to 20% on a rotated quartic.
- `examples/32_neural_particle_automata.ipynb` --- a growing neural cellular automaton whose cells are particles rather than pixels, after [Neural Particle Automata](https://arxiv.org/abs/2601.16096).
  Each cell carries a position as well as a state and the rule moves both, so the neighborhood is something the automaton decides as it goes rather than something the lattice fixes.
- `SPHPerceive` takes `fused`, which computes the same sums with Pallas kernels instead of array operations.
  The four quantities a particle perceives are four sums over the same pairs, and as array operations each one walks an array with an entry per pair; accumulated together in registers they do not.
  A kernel is opaque to automatic differentiation, so its derivative is worked out by hand and checked against the array version at every size, including sizes the tiling does not divide.
  It is GPU-only, and worth about five times the array path on a perception at a few thousand particles --- more on the backward than the forward, which is where the array version builds its largest intermediates.
  What it really buys is space, an entry per particle rather than per pair, so a cloud can grow past the size the array route cannot fit at all.

  Nothing about it is specialized to a cloud size.
  The tile shape is chosen from the particle count, and a count the tiles do not divide is padded and masked, the way an attention kernel handles a sequence length that is not a multiple of its block.
  The backward runs twice --- a particle's volume scales how every neighbor reads it, so that cotangent has to be complete before the positions can be resolved --- and each pass computes only the cotangents it uses rather than all of them.
- `SPHPerceive` gathers a particle's neighborhood by smoothed particle hydrodynamics: sums over whatever lies within a radius, weighted by a kernel that falls smoothly to zero at the edge, which is what keeps it differentiable while neighbors come and go.
  It reports the state, the neighborhood average, the state gradient and the density gradient --- the last having no counterpart on a lattice, where cells are evenly spaced by construction.
- `get_emoji_array` fetches an emoji already resized, scaled to the unit interval and framed in transparency.
  Five notebooks each carried their own copy of it.
- `examples/63_growing_nca_rl.ipynb` — a Growing NCA trained as a policy rather than by backpropagating through its whole development.
  The system is rolled out for half of its 128 steps and everything past that horizon is summarized by a learned value function, so cost scales with the horizon rather than with the length of the process.
  The automaton is differentiable, so the gradient is taken analytically through the rollout instead of estimated.

### Changed

- `examples/46_texture_nca.ipynb` matches feature distributions by sliced optimal transport rather than by Gram matrices, as the reference implementation does.
  Gram matrices give washed-out colour, which survived the fix above: the structure came back but the palette did not.
- The optimisation examples are renumbered to 60-66, making room in the neural cellular
  automata block for the two examples added above.
- Emoji glyphs are fetched from a pinned revision over a CDN rather than from a branch of the Noto Emoji repository.
  An unpinned URL made CAX's behaviour depend on the current state of another project's default branch, where a rename would have broken every installed version at once.
- `*Params` classes are now frozen dataclass pytrees, matching the `*State` convention: values the caller owns — immutable, and safe to pass through any JAX transformation.
  Previously they were NNX objects, whose reference semantics let a system share the caller's object and expose it to the write-back that NNX transforms perform by design; 0.4.2 patched that at the boundary, this removes the boundary.
  The unexported `cax.utils.numerics.detach` workaround is deleted.

  Construction is unchanged.
  The one visible difference is immutability: assigning to a field of a params object now raises `FrozenInstanceError`; build a new one with `dataclasses.replace` instead.
  `ParticleLeniaRuleParams.c_rep` remains static (part of the structure, not a leaf), as before.

### Fixed

- `examples/46_texture_nca.ipynb` matches its target.
  Its `gram_matrix` summed over the wrong einsum index and returned an array that kept a spatial position instead of correlating channel against channel, so the quantity being minimised was not a texture statistic and training barely moved: the loss fell from 7604 to 5402 over 2000 steps and the result looked nothing like the target.
  Nothing raised, because the shapes still matched between source and target.
- `get_emoji` accepts emoji written with more than one codepoint.
  It built its filename with `ord`, which raises on any sequence, so joined glyphs like 👨‍💻, skin tone modifiers like 👍🏽, and even ❤️ — a heart followed by a variation selector — all failed.
  Names are now spelled from every codepoint, dropping only the selector that asks for an emoji presentation, which Noto omits from its filenames.
  Flags raise a `ValueError` explaining that Noto stores them separately, rather than failing as a download error.

## 0.4.2

### Fixed

- `Lenia` and `ParticleLenia` no longer alias the caller's parameter objects.
  They stored `rule_params.growth_params` and `rule_params.kernel_params` directly, and because Flax writes a module's state back after a transformed call, constructing a system inside `jax.jit` or `jax.grad` wrote tracers into the caller's own `rule_params` — silently replacing its arrays with dead tracers, which then surfaced as an `UnexpectedTracerError` in a later, unrelated transformation.
  Systems now copy the containers, sharing the leaf arrays, via the new `cax.utils.numerics.detach`.

  This only affected building a system inside a transformation, which is the usual pattern when differentiating with respect to rule parameters; a system built once and called normally was never impacted.

## 0.4.1

A documentation and repository release: no API changes.

### Added

- `examples/65_lenia_grad.ipynb` — gradient descent through Lenia.
  Optimizes a creature to travel further, grows a target image from a rule, and finds the smallest perturbation that kills a soliton.
  Its opening section maps what a gradient reaches in a Lenia rule: the channel wiring is integer and cannot be differentiated, normalized parameters are flat along their scale direction, and the kernel radius is differentiable exactly when the kernel core vanishes at its support boundary — which the canonical exponential core does and the Gaussian core does not.
- `examples/66_lenia_grad_in_depth.ipynb` — the analysis companion: validating a
  gradient against finite differences, measuring the usable rollout length from a
  Lyapunov exponent, the two ways the gradient goes blind, the non-identifiability
  of `free_kernel_fn`, and what sharing a growth budget across channels costs.
- `tests/test_cs/test_lenia_grad.py` — the differentiability facts those notebooks
  rest on, as tests.

### Changed

- Notebooks are committed without outputs, and a `notebooks` CI job enforces it.
  Executed notebooks embed base64 media that does not delta-compress, so re-running one added its full size to the repository permanently; `examples/` drops from 59.5 MB to 0.4 MB and the source distribution shrinks with it.
  Example pages in the documentation now show code without stored results.

## 0.4.0

The first release since 0.3.3, covering both a backlog of unreleased work and a library-wide review.
It renames and removes public API — see **Migrating** below.

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
| `pip install cax[dev]` | dev tooling is a dependency group; `uv sync --dev` |

Three changes alter numerical results without raising:

- `FlowLenia`'s `theta_A` now defaults to `channel_size`, reproducing the official
  implementation for multi-channel systems (it previously defaulted to `1.0`).
- `ParticleLife.render` now maps simulation-y-up to image-y-down, matching `Boids`;
  renders are flipped vertically relative to 0.3.3.
- `FlowLenia`'s Sobel gradients now wrap the periodic domain (see below).

### Added

- Flow Lenia works in any number of spatial dimensions; the 2D-only restriction is gone.
  Gradients use n-dimensional separable Sobel kernels and reintegration tracking generalizes to `(2 * dd + 1) ** num_spatial_dims` displacements.
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
- Flow Lenia's Sobel gradients wrap the torus, consistent with its perception and reintegration.
  The official implementation zero-pads them, contradicting its own periodic domain; away from the boundary the two agree to float32 rounding.
  All Sobel gradients are computed by a single grouped convolution.
- Every constructor is keyword-only.
- Simulation states are frozen dataclasses registered as JAX pytrees, replaced with
  `dataclasses.replace` rather than mutated.
- Particle Lenia's default growth function is reference-faithful (`peak_growth_fn`);
  the previous default inflated the growth width by √2.
- Lenia kernel weights are normalized to sum to one, Chan's multi-kernel convention.
- `KernelParams.b` is `beta`.
- Perceive modules use an optimized pad-slice implementation, and Reaction-Diffusion's
  stencil is a plain constant rather than trainable weights.
- Development tooling moved from the `dev` extra to a PEP 735 dependency group, so `pytest`, `ruff`, and `ty` no longer ship in the published package metadata.
  Contributors keep using `uv sync --all-extras --dev`; `pip install cax[dev]` is gone.
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
