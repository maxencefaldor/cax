"""Elementary Cellular Automata update module."""

import jax
import jax.numpy as jnp

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ElementaryUpdate(Update):
	"""Elementary Cellular Automata update class."""

	patterns: jax.Array
	values: jax.Array

	def __init__(self, wolfram_code: str = "01101110"):
		"""Initialize the ElementaryUpdate.

		Args:
			wolfram_code: A string of 8 bits representing the Wolfram code for the automaton.

		"""
		self.patterns = jnp.array(
			[
				[1.0, 1.0, 1.0],
				[1.0, 1.0, 0.0],
				[1.0, 0.0, 1.0],
				[1.0, 0.0, 0.0],
				[0.0, 1.0, 1.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 1.0],
				[0.0, 0.0, 0.0],
			]
		)
		self.values = jnp.array([int(bit) for bit in wolfram_code])

	def __call__(self, state: State, perception: Perception, input: Input) -> State:
		"""Apply the Elementary Cellular Automata update rule.

		Args:
			state: The current state of the cellular automaton.
			perception: The perceived state of the neighborhood.
			input: Additional input to the update rule (not used in this implementation).

		Returns:
			The updated state of the cellular automaton.

		"""

		def update_pattern(pattern, value):
			return jnp.where(jnp.all(perception == pattern, axis=-1, keepdims=True), value, 0.0)

		state = jnp.sum(jax.vmap(update_pattern)(self.patterns, self.values), axis=0)
		return state
