"""Life perceive module.

This module implements the perception function for Conway's Game of Life and Life-like
cellular automata. It extracts each cell's state and the count of alive neighbors in
its Moore neighborhood (8 surrounding cells).
"""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import ConvPerceive, identity_kernel, neighbors_kernel


class LifePerceive(ConvPerceive):
	"""Life perception.

	Extracts each cell's current state and the count of alive neighbors in its Moore
	neighborhood. The perception consists of two channels: the cell's own state and
	the sum of its 8 neighboring cells. Uses circular padding for periodic boundaries.
	"""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize Life perceive.

		Args:
			rngs: rng key.

		"""
		channel_size = 1
		super().__init__(
			channel_size=channel_size,
			perception_size=2 * channel_size,
			kernel_size=(3, 3),
			padding="CIRCULAR",
			feature_group_count=channel_size,
			rngs=rngs,
		)

		kernel = jnp.concatenate([identity_kernel(2), neighbors_kernel(2)], axis=-1)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel[...] = kernel
