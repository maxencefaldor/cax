# Contributing to CAX

Thank you for your interest in contributing to CAX! We deeply appreciate you taking the time to help make CAX better. Whether you're contributing code, suggesting new features, opening an issue, improving documentation or writing tutorials - all contributions are valuable and welcome.

We also appreciate if you spread the word, for instance by starring the CAX GitHub repository, or referencing CAX in projects that used it.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the CAX repository by clicking the Fork button on the repository page. This creates a copy of the CAX repository in your own account.

2. Install Python >= 3.10 locally in order to run tests.

3. `pip` installing your fork from source. This allows you to modify the code and immediately test it out:

```bash
git clone https://github.com/maxencefaldor/cax
cd cax
pip install -e ".[dev]"  # Installs CAX from the current directory in editable mode.
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
ruff format .
ruff check .
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

- Every CA in CAX inherits from `nnx.Module` and follows the perceive/update architecture
- The perceive module defines how cells observe their neighborhood (e.g., `ConvPerceive`)
- The update module specifies how cells update their state based on these observations (e.g., `ResidualUpdate`, `NCAUpdate`, `LeniaUpdate`)

### Best Practices

1. **Vectorization**: Use JAX's `vmap` for operations applied to all cells
2. **Hardware Acceleration**: Leverage Flax components (e.g., `nnx.Conv`, `nnx.Linear`) rather than writing custom operations
3. **Batching**: Design your CA to handle batched inputs from the start
4. **JIT Compilation**: Ensure your CA is compatible with `jit` by avoiding Python control flow
5. **Random Number Handling**: Use `nnx.Rngs` for managing random states consistently

### Example Structure

You should design your perceive and update module, so that they are readily compatible with the core `CA` class.

```python
perceive = MyPerceive(...)
update = MyUpdate(...)

ca = CA(perceive, update)
```

A CA step will correspond to:

```python
@nnx.jit
def step(self, state: State, input: Input | None = None) -> State:
	"""Perform a single step of the CA.

	Args:
		state: Current state.
		input: Optional input.

	Returns:
		Updated state.

	"""
	perception = self.perceive(state)
	state = self.update(state, perception, input)
	return state
```

and a full forward pass will correspond to a simple jax.lax.scan of this function.

### Training Loop

For neural cellular automata training, a typical training loop follows this structure:

```python
@nnx.jit
def train_step(ca, optimizer, state, target, key):
	# Forward pass with value and gradient computation
	(loss, state), grad = nnx.value_and_grad(loss_fn, has_aux=True)(
		ca, state, target
	)
	# Update model parameters
	optimizer.update(grad)
	return loss, state

# Main training loop
for i in range(num_train_steps):
	key, subkey = jax.random.split(key)
	loss, state = train_step(ca, optimizer, state, target, subkey)
```

### Common Pitfalls

- Avoid Python loops over cells - use vectorized operations
- Don't mix NumPy and JAX arrays
- Keep track of random key usage for stochastic updates

For an extensive list of common gotchas in JAX, please read [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

CAX uses Flax NNX API, please read the [documentation](https://flax.readthedocs.io/en/latest/).

## License

By submitting a contribution to CAX, you agree to license your work under the same MIT License that covers the project. This helps keep the codebase open and accessible to everyone. If you have any questions about the licensing terms, please don't hesitate to reach out to the maintainers.