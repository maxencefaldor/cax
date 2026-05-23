"""Boids state module."""

from flax import nnx
from jax import Array


@nnx.dataclass
class BoidsState(nnx.Pytree):
	"""Boids state class."""

	position: Array = nnx.data()
	velocity: Array = nnx.data()
