"""Boids state."""

from flax import struct
from jax import Array


@struct.dataclass
class State:
	"""State for Boids."""

	position: Array  # (num_boids, num_spatial_dims)
	velocity: Array  # (num_boids, num_spatial_dims)
