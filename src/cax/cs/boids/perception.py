"""Boids perception module."""

from flax import struct
from jax import Array

from cax.core.perceive import Perception


@struct.dataclass
class BoidsPerception(Perception):
	"""Boids perception class."""

	acceleration: Array  # (num_boids, num_spatial_dims)
