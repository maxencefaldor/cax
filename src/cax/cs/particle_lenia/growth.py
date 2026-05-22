"""Particle Lenia growth module."""

from flax import nnx
from jax import Array


class GrowthParams(nnx.Pytree):
	"""Growth parameters."""

	def __init__(self, mean: Array, std: Array):
		"""Initialize growth parameters."""
		self.mean = mean
		self.std = std
