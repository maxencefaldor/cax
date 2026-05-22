"""Langton's Ant state module."""

from flax import nnx
from jax import Array


class LangtonAntState(nnx.Pytree):
	"""Langton's Ant state.

	Combines a 2D grid of cell colors with the ant's position and heading direction.

	Attributes:
		grid: Array of shape (height, width, 1) containing cell colors as float values
			in [0, num_colors). Each cell stores its current color index.
		position: Array of shape (2,) containing the ant's (row, col) position on the grid.
		direction: Scalar array containing the ant's heading as an integer in {0, 1, 2, 3}
			corresponding to {North, East, South, West}.

	"""

	def __init__(self, grid: Array, position: Array, direction: Array):
		"""Initialize Langton's Ant state."""
		self.grid = grid
		self.position = position
		self.direction = direction
