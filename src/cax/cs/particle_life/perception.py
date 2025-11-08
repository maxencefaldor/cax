"""Particle Life perception module."""

from flax import struct
from jax import Array


@struct.dataclass
class ParticleLifePerception:
	"""Particle Life perception class."""

	acceleration: Array  # (num_particles, num_spatial_dims)
