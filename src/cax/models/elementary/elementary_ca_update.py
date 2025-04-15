"""Elementary Cellular Automata update module."""

import jax
import jax.numpy as jnp
from cax.core.update.update import Update
from cax.types import Input, Perception, State
from flax import nnx
from jax import Array


class ElementaryCAUpdate(Update):
	"""Elementary Cellular Automata update class."""

	def __init__(self, *, wolfram_code: str = "01101110"):
		"""Initialize the ElementaryUpdate.

		Args:
			wolfram_code: A string of 8 bits representing the Wolfram code for the automaton.

		"""
		self.patterns = nnx.Param(
			jnp.array(
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
		)
		self.values = nnx.Param(jnp.array([int(bit) for bit in wolfram_code]))

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

		state = jnp.sum(jax.vmap(update_pattern)(self.patterns.value, self.values.value), axis=0)
		return state
