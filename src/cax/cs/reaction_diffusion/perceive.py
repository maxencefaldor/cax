"""Reaction-Diffusion perceive module.

This module implements the perception function for the Gray-Scott reaction-diffusion system.
It computes the discrete Laplacian of each chemical species using a convolution kernel,
which represents the spatial diffusion component of the PDE.
"""

import jax.numpy as jnp
from flax import nnx

from cax.core.perceive import ConvPerceive, grad2_kernel, identity_kernel


class ReactionDiffusionPerceive(ConvPerceive):
	"""Reaction-Diffusion perception.

	Computes the identity (current concentration) and the discrete Laplacian for each
	chemical species. The state has 2 channels (species U and V), so the perception
	produces 4 channels: [U, lap(U), V, lap(V)] via grouped convolution.
	"""

	def __init__(self, *, num_spatial_dims: int = 2, rngs: nnx.Rngs):
		"""Initialize Reaction-Diffusion perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions (default 2).
			rngs: RNG key.

		"""
		channel_size = 2
		kernel_size = tuple(3 for _ in range(num_spatial_dims))
		super().__init__(
			channel_size=channel_size,
			perception_size=2 * channel_size,
			kernel_size=kernel_size,
			padding="CIRCULAR",
			feature_group_count=channel_size,
			rngs=rngs,
		)

		kernel = jnp.concatenate(
			[
				identity_kernel(num_dims=num_spatial_dims),
				grad2_kernel(num_dims=num_spatial_dims, normalize=False),
			]
			* channel_size,
			axis=-1,
		)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel[...] = kernel
