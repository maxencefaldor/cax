# Contributing to CAX

Thank you for your interest in contributing to CAX! We deeply appreciate you taking the time to help make CAX better. Whether you're contributing code, suggesting new features, opening an issue, improving documentation or writing tutorials - all contributions are valuable and welcome.

We also appreciate if you spread the word, for instance by starring the CAX GitHub repository, or referencing CAX in projects that used it.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the CAX repository by clicking the Fork button on the repository page. This creates a copy of the CAX repository in your own account.

2. Clone your fork and go at the root of the repository.

3. Install your fork from source using [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

4. Add the CAX repository as an upstream remote, so you can use it to sync your changes.

```bash
git remote add upstream https://github.com/maxencefaldor/cax
```

5. Create a branch where you will develop from:

```bash
git checkout -b name-of-change
```

And implement your changes using your favorite editor.

6. Make sure your code passes CAX’s lint and type checks, by running the following from the top of the repository:

```bash
uv ruff check .   # Linting
uv ruff format .  # Formatting
```

7. Make sure the tests pass by running the following command from the top of the repository:

```bash
pytest tests/
```

8. Once you are satisfied with your change, create a commit as follows ( how to write a commit message):

```bash
git add file1.py file2.py ...
git commit -m "Your commit message"
```

Then sync your code with the main repo:

```bash
git fetch upstream
git rebase upstream/main
```

Finally, push your commit on your development branch and create a remote branch in your fork that you can use to create a pull request from:

```bash
git push --set-upstream origin name-of-change
```

9. Create a pull request from the CAX repository and send it for review.

## Report a bug or suggest a new feature using GitHub issues

Go to [https://github.com/maxencefaldor/cax/issues](https://github.com/maxencefaldor/cax/issues) and click on "New issue".

Informative bug reports tend to have:

- A quick summary
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Additional notes

## Designing Efficient CAX Architectures

### Core Principles

- Every complex system in CAX inherits from `nnx.Module` and follows the perceive/update architecture
- The perceive module defines how cells observe their neighborhood (e.g., `ConvPerceive`)
- The update module specifies how cells update their state based on these observations (e.g., `ResidualUpdate`, `NCAUpdate`, `LeniaUpdate`)

### Best Practices

1. **Vectorization**: Use JAX's `vmap` for operations applied to all cells
2. **Hardware Acceleration**: Leverage Flax components (e.g., `nnx.Conv`, `nnx.Linear`) rather than writing custom operations
3. **Batching**: Design your complex system to handle batched inputs from the start
4. **JIT Compilation**: Ensure your complex system is compatible with `jit` by avoiding Python control flow
5. **Random Number Handling**: Use `nnx.Rngs` for managing random states consistently

### Example Structure

In CAX, every complex system must inherit from the `ComplexSystem` class and implement two required methods:
- `_step`: Defines how the system evolves over one time step
- `render`: Converts the system state into a visual representation

The `_step` method can perform any computation, but it must follow this signature: take a state as input, an optional input, and return an updated state. Many complex systems (like cellular automata or particle systems) follow a common pattern where individual components (e.g., cells, particles, etc.) first perceive their local neighborhood, then update their state based on this perception and current state. For this reason, we recommend structuring the `_step` method into two phases:

1. **Perceive**: Gather information from the neighborhood
2. **Update**: Modify the state based on current state and perception

This structure is optional but helps organize the code clearly.

You should design your perceive and update modules so that they are readily compatible with the core `ComplexSystem` class.

```python
class CustomNCA(ComplexSystem[Array, Array]):
	"""Custom neural cellular automaton."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize custom cellular automaton.

		Args:
			rngs: rngs key.

		"""
		# CAX provides a set of perceive modules but you can define your own.
		self.perceive = CustomPerceive(...)

		# CAX provides a set of update modules but you can define your own.
		self.update = CustomUpdate(...)

	def _step(self, state: Array, input: Array | None = None) -> Array:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)
		return next_state

	@nnx.jit
	def render(self, state: Array) -> Array:
		"""Render state to RGB."""
		rgba = state[..., -4:]
		rgb = rgba_to_rgb(rgba)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
```

### Library Conventions

These are the decisions the codebase holds everywhere; new code follows them.

- **State typing is binary: trainable if and only if `nnx.Param`.** Learned weights are
  `nnx.Param`; rule tables, physics stencils, derived caches, and every other array are
  plain data. The two filters `nnx.state(cs, nnx.Param)` and
  `nnx.state(cs, nnx.Not(nnx.Param))` then mean exactly "the weights" and "everything
  else" — which is what optimizers, checkpointers, and parameter-space searches need.
  Storing a fixed quantity as a `Param` puts physics where optimizers look.
- **Simulation states are frozen dataclasses registered as JAX pytrees**
  (`@jax.tree_util.register_dataclass` over `@dataclass(frozen=True)`), replaced with
  `dataclasses.replace`, never mutated. States are values: mutating one aliases the
  caller's object, and `nnx.scan` carries graph nodes by reference. Modules hold the
  long-lived state; states flow through them.
- **Trajectories are scan outputs.** `__call__` returns the final state; under
  `return_states=True` it returns `(final_state, states)` with the per-step states
  stacked as the scan's outputs, mirroring `jax.lax.scan`'s `(carry, ys)`. Nothing is sown onto the module
  by the drivers. Per-step metrics are functions applied to the returned trajectory.
  `sow` is reserved for genuinely optional internals and is only ever called inside
  an active `nnx.capture`, named after the value at the sow site.
- **Gradient safety is input sanitization.** Any division, norm, or singular kernel
  evaluated where its argument can degenerate goes through
  `cax.utils.safe_divide` / `safe_norm` or repeats their double-`where` pattern —
  masking an invalid output after computing it leaves `nan` in the gradient even when
  the forward pass is finite.
- **Randomness at call time draws from a named stream** (e.g. `rngs.noise()`), never
  from `params`, which is the initialization stream; a dedicated name keeps simulation
  noise independent of how many parameters were initialized and makes
  `nnx.split_rngs(..., only=...)` filtering correct. Constructors take keyword-only
  `*, rngs: nnx.Rngs` exactly when the system draws randomness — at initialization
  or at call time; a deterministic system takes none. A required argument that does
  nothing misstates what the system is.
- **Validation raises `ValueError` for anything a user can trigger**; `assert` is
  reserved for internal invariants (and `python -O` removes it).
- **Every module that cites `[n]` carries its own `References:` block** in its module
  docstring; entries give the work and, where useful, the URL.
- **Symbols follow the cited reference's equations.** A single-letter or Greek
  parameter name (`R`, `T`, `beta`, `theta_A`) is kept exactly when the system's
  canonical reference uses that symbol in its equations — reading the code against
  the paper is the use case — and is spelled out otherwise (`attraction_matrix`,
  `feed_rate`). The docstring expands every kept symbol. This is why `T` exists in
  the Lenia family while Gray-Scott keeps `dt`: each system reads like its own
  reference, and that rule — not surface-identical names — is the library's
  uniformity.
- **`spatial_dims` is a shape; `num_spatial_dims` is a rank.** Grid systems take
  the full spatial shape — Lenia's FFT kernels are precomputed at grid size — while
  particle systems take only the dimensionality. The similar names carry a real
  semantic difference and are deliberately not unified.
- **Reference fidelity is load-bearing.** Each system mirrors its reference
  implementation's formulas exactly — including where references disagree with each
  other (grid Lenia's Gaussian carries a 1/2 factor, Particle Lenia's does not).
  Deviations are bugs unless the docstring states them as decisions.

### Common Pitfalls

- Avoid Python loops over cells - use vectorized operations
- Don't mix NumPy and JAX arrays
- Keep track of random key usage for stochastic updates

For an extensive list of common gotchas in JAX, please read [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

CAX uses Flax NNX API, please read the [documentation](https://flax.readthedocs.io/en/latest/).

## License

By submitting a contribution to CAX, you agree to license your work under the same MIT License that covers the project. This helps keep the codebase open and accessible to everyone. If you have any questions about the licensing terms, please don't hesitate to reach out to the maintainers.
