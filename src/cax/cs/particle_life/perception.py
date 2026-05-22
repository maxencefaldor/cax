"""Particle Life perception module."""

from flax import nnx
from jax import Array


class ParticleLifePerception(nnx.Pytree):
	"""Particle Life perception class."""

	def __init__(self, acceleration: Array):
		"""Initialize Particle Life perception."""
		self.acceleration = acceleration  # (num_particles, num_spatial_dims)
