"""Life update module."""

import jax.numpy as jnp

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class LifeUpdate(Update):
	"""Life update class."""

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Game of Life rules to update the cellular automaton state.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		# Extract cell state and neighbor count from perception
		self_alive = perception[..., 0:1]
		neighbors_alive_count = perception[..., 1:2]

		# Conway's Game of Life rules:
		# 1. Any live cell with two or three live neighbours survives.
		# 2. Any dead cell with three live neighbours becomes a live cell.
		# 3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.

		# Implement the rules
		stay_alive = jnp.logical_and(
			self_alive == 1, jnp.logical_or(neighbors_alive_count == 2, neighbors_alive_count == 3)
		)
		become_alive = jnp.logical_and(self_alive == 0, neighbors_alive_count == 3)

		# Combine the conditions
		state = jnp.logical_or(stay_alive, become_alive).astype(state.dtype)
		return state
