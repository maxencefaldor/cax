"""Particle Life types."""

from flax import struct
from jax import Array


@struct.dataclass
class State:
	"""State for Particle Life."""

	class_: Array  # (num_particles,)
	position: Array  # (num_particles, num_dims)
	velocity: Array  # (num_particles, num_dims)


@struct.dataclass
class Perception:
	"""Perception for Particle Life."""

	acceleration: Array  # (num_particles, num_dims)
