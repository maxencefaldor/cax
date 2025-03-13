"""Boids model."""

import jax.numpy as jnp
from cax.core.ca import CA
from flax import nnx

from .boids_perceive import BoidsPerceive
from .boids_update import BoidsUpdate


class Boids(CA):
	"""Boids model."""

	def __init__(
		self,
		boid_policy: nnx.Module,
		*,
		dt: float = 0.01,
		velocity_half_life: float = jnp.inf,
		boundary: str = "CIRCULAR",
	):
		"""Initialize Boids."""
		perceive = BoidsPerceive(
			boid_policy=boid_policy,
		)

		update = BoidsUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
			boundary=boundary,
		)

		super().__init__(perceive, update)
