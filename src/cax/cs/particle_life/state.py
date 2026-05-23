"""Particle Life state module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class ParticleLifeState(nnx.Pytree):
	"""Particle Life state class."""

	class_: Array
	position: Array
	velocity: Array
