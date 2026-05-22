"""Langton's Ant state module."""

from flax import struct
from jax import Array

from cax.core import State


@struct.dataclass
class LangtonAntState(State):
	"""Langton's Ant state.

	Combines a 2D grid of cell colors with the ant's position and heading direction.

	Attributes:
		grid: Array of shape (height, width, 1) containing cell colors as float values
			in [0, num_colors). Each cell stores its current color index.
		position: Array of shape (2,) containing the ant's (row, col) position on the grid.
		direction: Scalar array containing the ant's heading as an integer in {0, 1, 2, 3}
			corresponding to {North, East, South, West}.

	"""

	grid: Array
	position: Array
	direction: Array
