"""Life model."""

import jax
import jax.numpy as jnp
from cax.core.ca import CA
from cax.types import State
from cax.utils.render import clip_and_uint8
from flax import nnx

from .life_perceive import LifePerceive
from .life_update import LifeUpdate


class Life(CA):
	"""Life model."""

	def __init__(self, rngs: nnx.Rngs):
		"""Initialize Life."""
		perceive = LifePerceive(rngs=rngs)
		update = LifeUpdate()

		super().__init__(perceive, update)

	@nnx.jit
	def render(self, state: State) -> jax.Array:
		"""Render state.

		Args:
			state: An array of states.

		Returns:
			Rendered states.

		"""
		frame = jnp.repeat(state, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(frame)
