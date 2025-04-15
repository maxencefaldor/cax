"""Life perceive module."""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import ConvPerceive, identity_kernel, neighbors_kernel


class LifePerceive(ConvPerceive):
	"""Life perceive class."""

	def __init__(self, rngs: nnx.Rngs, *, padding: str = "CIRCULAR"):
		"""Initialize LifePerceive."""
		channel_size = 1
		super().__init__(
			channel_size=channel_size,
			perception_size=2 * channel_size,
			rngs=rngs,
			kernel_size=(3, 3),
			padding=padding,
			feature_group_count=channel_size,
		)

		kernel = jnp.concatenate([identity_kernel(2), neighbors_kernel(2)], axis=-1)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel = nnx.Param(kernel)
