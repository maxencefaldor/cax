"""Life perceive module.

This module implements the perception function for Conway's Game of Life and Life-like
cellular automata. It extracts each cell's state and the count of alive neighbors in
its Moore neighborhood (8 surrounding cells).
"""

import jax.numpy as jnp

from cax.core.perceive import MoorePerceive


class LifePerceive(MoorePerceive):
	"""Life perception.

	Extracts each cell's current state and the count of alive neighbors in its Moore
	neighborhood. The perception consists of two channels: the cell's own state and
	the sum of its 8 neighboring cells. Uses circular padding for periodic boundaries.
	"""

	def __init__(self):
		"""Initialize Life perceive."""
		super().__init__(num_spatial_dims=2, radius=1, padding="CIRCULAR", reduce_fn=jnp.sum)
