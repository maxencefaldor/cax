"""Boids perception module."""

from flax import struct
from jax import Array


@struct.dataclass
class BoidsPerception:
	"""Boids perception class."""

	acceleration: Array  # (num_boids, num_spatial_dims)
