"""Elementary Cellular Automata update module.

This module implements the update rule for Elementary Cellular Automata based on Wolfram codes.
Each cell's next state is determined by looking up its three-cell neighborhood configuration
in the Wolfram code lookup table.

"""

import jax
import jax.numpy as jnp
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update


class ElementaryUpdate(Update):
	"""Elementary Cellular Automata update rule.

	Applies the Wolfram rule by matching each cell's three-cell neighborhood against all
	possible configurations and selecting the corresponding output value from the Wolfram code.

	"""

	def __init__(self, *, wolfram_code: Array):
		"""Initialize Elementary update.

		Args:
			wolfram_code: Array of 8 binary values defining the Wolfram rule. Each element
				corresponds to the output for one of the 8 possible three-cell neighborhood
				configurations (111, 110, 101, 100, 011, 010, 001, 000).

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
		"""Process the current state, perception, and input to produce a new state.

		Matches each cell's three-cell neighborhood configuration against the Wolfram code
		lookup table to determine the next state.

		Args:
			state: Current state (unused, next state computed solely from perception).
			perception: Array with shape (..., width, 3) containing the three-cell neighborhood
				(left, self, right) for each cell.
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (..., width, 1) containing the updated cell values.

		"""

		def update_pattern(pattern: Array, value: Array) -> Array:
			return jnp.where(jnp.all(perception == pattern, axis=-1, keepdims=True), value, 0.0)

		state = jnp.sum(jax.vmap(update_pattern)(self.configurations, self.wolfram_code), axis=0)
		return state
