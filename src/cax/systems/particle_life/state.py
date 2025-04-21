"""Particle Life state."""

from flax import struct
from jax import Array


@struct.dataclass
class State:
	"""State for Particle Life."""

	class_: Array  # (num_particles,)
	position: Array  # (num_particles, num_spatial_dims)
	velocity: Array  # (num_particles, num_spatial_dims)
