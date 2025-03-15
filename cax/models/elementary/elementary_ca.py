"""Elementary Cellular Automata model."""

import jax
import jax.numpy as jnp
from cax.core.ca import CA
from cax.types import State
from cax.utils.render import clip_and_uint8
from flax import nnx

from .elementary_ca_perceive import ElementaryCAPerceive
from .elementary_ca_update import ElementaryCAUpdate


class ElementaryCA(CA):
	"""Elementary Cellular Automata model."""

	def __init__(self, rngs: nnx.Rngs, wolfram_code: str = "01101110"):
		"""Initialize Elementary CA."""
		perceive = ElementaryCAPerceive(rngs=rngs)
		update = ElementaryCAUpdate(wolfram_code)

		super().__init__(perceive, update)

	@nnx.jit
	def render(self, state: State) -> jax.Array:
		"""Render states as a space-time diagram.

		Args:
			state: An array of successive states.

		Returns:
			Rendered states as a space-time diagram.

		"""
		frame = jnp.repeat(state, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(frame)
