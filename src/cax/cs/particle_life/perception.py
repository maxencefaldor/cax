"""Particle Life perception module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class ParticleLifePerception(nnx.Pytree):
	"""Particle Life perception class."""

	acceleration: Array
