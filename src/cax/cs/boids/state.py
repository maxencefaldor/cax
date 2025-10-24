"""Boids state module."""

from flax import struct
from jax import Array

from cax.core import State


@struct.dataclass
class BoidsState(State):
	"""Boids state class."""

	position: Array  # (num_boids, num_spatial_dims)
	velocity: Array  # (num_boids, num_spatial_dims)
