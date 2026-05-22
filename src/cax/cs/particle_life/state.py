"""Particle Life state module."""

from flax import nnx
from jax import Array


class ParticleLifeState(nnx.Pytree):
	"""Particle Life state class."""

	def __init__(self, class_: Array, position: Array, velocity: Array):
		"""Initialize Particle Life state."""
		self.class_ = class_  # (num_particles,)
		self.position = position  # (num_particles, num_spatial_dims)
		self.velocity = velocity  # (num_particles, num_spatial_dims)
