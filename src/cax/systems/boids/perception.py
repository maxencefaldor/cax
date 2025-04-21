"""Boids perception."""

from flax import struct
from jax import Array


@struct.dataclass
class Perception:
	"""Perception for Boids."""

	acceleration: Array  # (num_particles, num_spatial_dims)
