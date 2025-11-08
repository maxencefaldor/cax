"""Life update module.

This module implements the update rule for Conway's Game of Life and Life-like cellular
automata based on birth/survival conditions. Cells become alive (birth) or stay alive
(survival) based on the number of alive neighbors.
"""

import jax.numpy as jnp
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update


class LifeUpdate(Update):
	"""Life update rule.

	Applies birth and survival rules to determine each cell's next state. Dead cells with
	the right number of alive neighbors are born, and alive cells with the right number
	of alive neighbors survive. All other cells become or remain dead.
	"""

	def __init__(self, birth: Array, survival: Array):
		"""Initialize Life update.

		Args:
			birth: Array of shape (9,) defining birth conditions. Element i is 1.0 if a dead
				cell with i alive neighbors should become alive, 0.0 otherwise.
			survival: Array of shape (9,) defining survival conditions. Element i is 1.0 if a
				live cell with i alive neighbors should stay alive, 0.0 otherwise.

		"""
		self.birth = birth
		self.survival = survival

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		Determines each cell's next state by checking birth conditions for dead cells and
		survival conditions for alive cells based on the number of alive neighbors.

		Args:
			state: Current state (unused, next state computed solely from perception).
			perception: Array with shape (..., height, width, 2) where channel 0 is the cell's
				state (0.0 or 1.0) and channel 1 is the count of alive neighbors (0-8).
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (..., height, width, 1) containing binary cell values.

		"""
		self_alive = perception[..., 0:1]
		num_alive_neighbors = perception[..., 1:2].astype(jnp.int32)

		# Birth
		birth = jnp.logical_and(1.0 - self_alive, self.birth[num_alive_neighbors])

		# Survival
		survival = jnp.logical_and(self_alive, self.survival[num_alive_neighbors])

		# Combine the conditions for the next state
		state = jnp.where(birth | survival, 1.0, 0.0)
		return state
