"""Particle Lenia growth module."""

from flax import struct
from jax import Array


@struct.dataclass
class GrowthParams:
	"""Growth parameters."""

	mean: Array
	std: Array
