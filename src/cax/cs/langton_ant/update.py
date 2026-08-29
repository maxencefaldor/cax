"""Langton's Ant update module.

This module implements the update rule for Langton's Ant and its multi-color generalizations.
The ant turns based on the current cell color, flips the cell to the next color, and moves
forward one step.
"""

from dataclasses import replace
from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.update import Update

from .state import LangtonAntState

DIRECTION_VECTORS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


class LangtonAntUpdate(Update[LangtonAntState, Array, Array]):
	"""Langton's Ant update rule.

	Applies the generalized Langton's Ant rule: the ant turns according to the current cell
	color, flips the cell to the next color in the cycle, and advances one step in its new
	heading direction. Boundaries wrap periodically.
	"""

	def __init__(self, *, turns: Array):
		"""Initialize Langton's Ant update.

		Args:
			turns: Array of shape (num_colors,) with integer values in {0, 1, 2, 3}
				encoding turn directions for each cell color. Values correspond to
				clockwise rotation in multiples of 90 degrees: 0=no turn, 1=right,
				2=u-turn, 3=left.

		"""
		self.turns = turns

	@override
	def __call__(
		self, state: LangtonAntState, perception: Array, input: Array | None = None
	) -> LangtonAntState:
		"""Process the current state and perception to produce the next state.

		Executes the Langton's Ant transition: turn, flip cell color, move forward.

		Args:
			state: Current Langton's Ant state.
			perception: Scalar array containing the cell color at the ant's position.
			input: Optional input (unused).

		Returns:
			Next Langton's Ant state with updated grid, position, and direction.

		"""
		num_colors = self.turns.shape[0]
		grid_height = state.grid.shape[0]
		grid_width = state.grid.shape[1]

		cell_color = perception.astype(jnp.int32)

		turn = self.turns[cell_color]
		new_direction = (state.direction + turn) % 4

		new_color = ((cell_color + 1) % num_colors).astype(state.grid.dtype)
		row, col = state.position[0], state.position[1]
		new_grid = state.grid.at[row, col, 0].set(new_color)

		delta = DIRECTION_VECTORS[new_direction]
		new_position = jnp.stack([(row + delta[0]) % grid_height, (col + delta[1]) % grid_width])

		return replace(state, grid=new_grid, position=new_position, direction=new_direction)
