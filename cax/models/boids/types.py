"""Boids types."""

from flax import struct
from jax import Array


@struct.dataclass
class State:
	"""State for Boids."""

	position: Array  # (num_boids, num_dims)
	velocity: Array  # (num_boids, num_dims)


@struct.dataclass
class Perception:
	"""Perception for Boids."""

	acceleration: Array  # (num_particles, num_dims)
