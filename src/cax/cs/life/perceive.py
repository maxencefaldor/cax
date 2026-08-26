"""Life perceive module.

This module implements the perception function for Conway's Game of Life and Life-like
cellular automata. It extracts each cell's state and the count of alive neighbors in
its Moore neighborhood (8 surrounding cells).
"""

from typing import Literal

import jax.numpy as jnp

from cax.core.perceive import MoorePerceive


class LifePerceive(MoorePerceive):
	"""Life perception.

	Extracts each cell's current state and the count of alive neighbors in its Moore
	neighborhood. The perception consists of two channels: the cell's own state and
	the sum of its 8 neighboring cells.
	"""

	def __init__(self, *, padding: Literal["CIRCULAR", "ZERO"] = "CIRCULAR"):
		"""Initialize Life perceive.

		Args:
			padding: Boundary condition mode. "CIRCULAR" for periodic boundaries,
				"ZERO" for a border of permanently dead cells.

		"""
		super().__init__(num_spatial_dims=2, radius=1, padding=padding, reduce_fn=jnp.sum)
