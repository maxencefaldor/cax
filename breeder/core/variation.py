"""Variation utilities.

Generic building blocks for mutation operators: bounded Gaussian perturbations
with reflective boundary handling.
"""

import jax
import jax.numpy as jnp
from jax import Array


def reflect(x: Array, *, lower: float | None = None, upper: float | None = None) -> Array:
	"""Reflect x into bounds using reflective boundary conditions.

	Supports one-sided reflection (only lower or only upper) and two-sided reflection
	(both lower and upper).

	Args:
		x: Array to reflect.
		lower: Lower bound. If None, no lower reflection is applied.
		upper: Upper bound. If None, no upper reflection is applied.

	Returns:
		Array reflected into the specified bounds.

	"""
	# Two-sided: periodic folding into [lower, upper]
	if lower is not None and upper is not None:
		span = upper - lower
		x = jnp.abs(x - lower)

		n = jnp.floor(x / span).astype(int)
		x = x - n * span
		x = jnp.where(n % 2 == 0, x, span - x)

		return x + lower

	# One-sided or no reflection
	if lower is not None:
		x = jnp.where(x < lower, 2 * lower - x, x)
	if upper is not None:
		x = jnp.where(x > upper, 2 * upper - x, x)
	return x


def mutate_bounded(
	key: Array,
	x: Array,
	*,
	lower: float,
	upper: float,
	mutation_std: float,
) -> Array:
	"""Gaussian mutation with scale relative to range, reflected into bounds.

	Args:
		key: rng key.
		x: Array to mutate.
		lower: Lower bound.
		upper: Upper bound.
		mutation_std: Standard deviation relative to the range `upper - lower`.

	Returns:
		Mutated array reflected into `[lower, upper]`.

	"""
	noise = mutation_std * (upper - lower) * jax.random.normal(key, x.shape)
	return reflect(x + noise, lower=lower, upper=upper)
