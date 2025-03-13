"""Particle Life model."""

import jax
from cax.core.ca import CA
from flax import nnx

from .particle_life_perceive import ParticleLifePerceive
from .particle_life_update import ParticleLifeUpdate


class ParticleLife(CA):
	"""Particle life model."""

	def __init__(
		self,
		num_classes: int,
		rngs: nnx.Rngs,
		*,
		r_max: float = 0.15,
		beta: float = 0.3,
		dt: float = 0.01,
		velocity_half_life: float = 0.01,
		force_factor: float = 1.0,
		boundary: str = "CIRCULAR",
	):
		"""Initialize Particle Life."""
		key = rngs.params()
		A = jax.random.uniform(key, (num_classes, num_classes), minval=-1.0, maxval=1.0)

		perceive = ParticleLifePerceive(
			A=A,
			r_max=r_max,
			beta=beta,
			force_factor=force_factor,
			boundary=boundary,
		)

		update = ParticleLifeUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
			boundary=boundary,
		)

		super().__init__(perceive, update)
