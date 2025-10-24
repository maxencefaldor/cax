"""Particle Life state module."""

from flax import struct
from jax import Array


@struct.dataclass
class ParticleLifeState:
	"""Particle Life state class."""

	class_: Array  # (num_particles,)
	position: Array  # (num_particles, num_spatial_dims)
	velocity: Array  # (num_particles, num_spatial_dims)
