"""Boids perception module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class BoidsPerception(nnx.Pytree):
	"""Boids perception class."""

	acceleration: Array = nnx.data()
