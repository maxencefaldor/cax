# Contributing to CAX

Thank you for your interest in contributing to CAX! We deeply appreciate you taking the time to help make CAX better. Whether you're contributing code, suggesting new features, opening an issue, improving documentation or writing tutorials - all contributions are valuable and welcome.

We also appreciate if you spread the word, for instance by starring the CAX GitHub repository, or referencing CAX in projects that used it.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the CAX repository by clicking the Fork button on the repository page. This creates a copy of the CAX repository in your own account.

2. Clone your fork and go at the root of the repository.

3. Install your fork from source using [https://docs.astral.sh/uv/](`uv`).

```bash
uv run python -c "import cax; print(cax.__doc__)"
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

Go to https://github.com/maxencefaldor/cax/issues and click on "New issue".

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

The `_step` method can perform any computation, but it must follow this signature: take a `State` as input, an optional `Input`, and return an updated `State`. Many complex systems (like cellular automata or particle systems) follow a common pattern where individual components (e.g., cells, particles, etc.) first perceive their local neighborhood, then update their state based on this perception and current state. For this reason, we recommend structuring the `_step` method into two phases:

1. **Perceive**: Gather information from the neighborhood
2. **Update**: Modify the state based on current state and perception

This structure is optional but helps organize the code clearly.

You should design your perceive and update modules so that they are readily compatible with the core `ComplexSystem` class.

```python
class CustomNCA(ComplexSystem):
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

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state):
		"""Render state to RGB."""
		rgba = state[..., -4:]
		rgb = rgba_to_rgb(rgba)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
```

### Common Pitfalls

- Avoid Python loops over cells - use vectorized operations
- Don't mix NumPy and JAX arrays
- Keep track of random key usage for stochastic updates

For an extensive list of common gotchas in JAX, please read [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

CAX uses Flax NNX API, please read the [documentation](https://flax.readthedocs.io/en/latest/).

## License

By submitting a contribution to CAX, you agree to license your work under the same MIT License that covers the project. This helps keep the codebase open and accessible to everyone. If you have any questions about the licensing terms, please don't hesitate to reach out to the maintainers.
