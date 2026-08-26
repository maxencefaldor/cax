"""Langton's Ant perceive module.

This module implements the perception function for Langton's Ant. It extracts the color
of the cell at the ant's current position, which determines the turn direction.
"""

from jax import Array

from cax.core.perceive import Perceive
from cax.core.perceive.perceive import Perception

from .state import LangtonAntState


class LangtonAntPerceive(Perceive[LangtonAntState]):
	"""Langton's Ant perception.

	Reads the cell color at the ant's current grid position. The perceived color is used
	by the update rule to determine the turn direction.
	"""

	def __call__(self, state: LangtonAntState) -> Perception:
		"""Perceive the cell color at the ant's position.

		Args:
			state: Current Langton's Ant state containing grid, position, and direction.

		Returns:
			Scalar array containing the cell color at the ant's position.

		"""
		row, col = state.position[0], state.position[1]
		cell_color: Array = state.grid[row, col, 0]
		return cell_color
