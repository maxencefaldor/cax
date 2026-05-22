"""Boids state module."""

from flax import nnx
from jax import Array


class BoidsState(nnx.Pytree):
	"""Boids state class."""

	def __init__(self, position: Array, velocity: Array):
		"""Initialize boids state."""
		self.position = position  # (num_boids, num_spatial_dims)
		self.velocity = velocity  # (num_boids, num_spatial_dims)
