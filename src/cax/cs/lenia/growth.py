"""Lenia growth module."""

from flax import nnx
from jax import Array

from .kernel import bell


@nnx.dataclass
class GrowthParams(nnx.Pytree):
	"""Growth parameters."""

	mean: Array
	std: Array


def exponential_growth_fn(u: Array, growth_params: GrowthParams) -> Array:
	"""Growth mapping function introduced in [1]."""
	return 2 * bell(u, growth_params.mean, growth_params.std) - 1
