"""Particle Life state module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class ParticleLifeState(nnx.Pytree):
	"""Particle Life state class."""

	class_: Array = nnx.data()
	position: Array = nnx.data()
	velocity: Array = nnx.data()
