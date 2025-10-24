"""Boids state module."""

from flax import struct
from jax import Array


@struct.dataclass
class BoidsState:
	"""Boids state class."""

	position: Array  # (num_boids, num_spatial_dims)
	velocity: Array  # (num_boids, num_spatial_dims)
