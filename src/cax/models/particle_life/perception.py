"""Particle Life perception."""

from flax import struct
from jax import Array


@struct.dataclass
class Perception:
	"""Perception for Particle Life."""

	acceleration: Array  # (num_particles, num_spatial_dims)
