"""Elementary Cellular Automata perceive module.

This module implements the perception function for Elementary Cellular Automata. It extracts
the three-cell neighborhood (left, self, right) for each cell using a fixed convolution kernel.
"""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import ConvPerceive


class ElementaryPerceive(ConvPerceive):
	"""Elementary Cellular Automata perception.

	Extracts the three-cell neighborhood for each cell in a one-dimensional cellular automaton.
	The perception consists of three channels: left neighbor, self, and right neighbor.
	Uses circular padding to implement periodic boundary conditions.
	"""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize Elementary perceive.

		Args:
			rngs: rng key.

		"""
		channel_size = 1
		super().__init__(
			channel_size=channel_size,
			perception_size=3 * channel_size,
			kernel_size=(3,),
			padding="CIRCULAR",
			feature_group_count=channel_size,
			use_bias=False,
			activation_fn=None,
			rngs=rngs,
		)

		left_kernel = jnp.array([[1.0], [0.0], [0.0]])
		identity_kernel = jnp.array([[0.0], [1.0], [0.0]])
		right_kernel = jnp.array([[0.0], [0.0], [1.0]])

		kernel = jnp.concatenate([left_kernel, identity_kernel, right_kernel], axis=-1)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel.value = kernel
