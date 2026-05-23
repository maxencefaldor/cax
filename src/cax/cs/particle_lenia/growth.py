"""Particle Lenia growth module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class GrowthParams(nnx.Pytree):
	"""Growth parameters."""

	mean: Array
	std: Array
