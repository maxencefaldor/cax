"""Elementary Cellular Automata module."""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from .perceive import ElementaryPerceive
from .update import ElementaryUpdate


class Elementary(ComplexSystem):
	"""Elementary Cellular Automata class."""

	def __init__(
		self,
		*,
		wolfram_code: Array,
		rngs: nnx.Rngs,
	):
		"""Initialize Elementary Cellular Automaton.

		Args:
			wolfram_code: The Wolfram code of the Elementary Cellular Automaton.
			rngs: rng key.

		"""
		self.perceive = ElementaryPerceive(rngs=rngs)
		self.update = ElementaryUpdate(wolfram_code=wolfram_code)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@classmethod
	def wolfram_code_from_rule_number(cls, rule_number: int) -> Array:
		"""Create Wolfram code array from a rule number.

		Args:
			rule_number: The rule number.

		Returns:
			The Wolfram code array.

		"""
		assert 0 <= rule_number < 256, "Wolfram code must be between 0 and 255."
		return ((rule_number >> 7 - jnp.arange(8)) & 1).astype(jnp.float32)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		return clip_and_uint8(rgb)
