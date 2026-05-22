"""Boids perception module."""

from flax import nnx
from jax import Array


class BoidsPerception(nnx.Pytree):
	"""Boids perception class."""

	def __init__(self, acceleration: Array):
		"""Initialize boids perception."""
		self.acceleration = acceleration  # (num_boids, num_spatial_dims)
