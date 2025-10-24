"""Elementary Cellular Automata update module."""

import jax
import jax.numpy as jnp
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ElementaryUpdate(Update):
	"""Elementary Cellular Automata update class."""

	def __init__(self, wolfram_code: Array):
		"""Initialize Elementary update.

		Args:
			wolfram_code: The Wolfram code of the Elementary Cellular Automaton.

		"""
		self.configurations = jnp.array(
			[
				[1, 1, 1],
				[1, 1, 0],
				[1, 0, 1],
				[1, 0, 0],
				[0, 1, 1],
				[0, 1, 0],
				[0, 0, 1],
				[0, 0, 0],
			],
			dtype=jnp.float32,
		)
		self.wolfram_code = wolfram_code

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Elementary Cellular Automata update rule.

		Args:
			state: The current state of the cellular automaton.
			perception: The perceived state of the neighborhood.
			input: Additional input to the update rule (not used in this implementation).

		Returns:
			The updated state of the cellular automaton.

		"""

		def update_pattern(pattern: Array, value: Array) -> Array:
			return jnp.where(jnp.all(perception == pattern, axis=-1, keepdims=True), value, 0.0)

		state = jnp.sum(jax.vmap(update_pattern)(self.configurations, self.wolfram_code), axis=0)
		return state
