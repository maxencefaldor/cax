"""Particle Life perception module."""

from flax import struct
from jax import Array

from cax.core.perceive import Perception


@struct.dataclass
class ParticleLifePerception(Perception):
	"""Particle Life perception class."""

	acceleration: Array  # (num_particles, num_spatial_dims)
