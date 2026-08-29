"""Reaction-Diffusion perceive module.

This module implements the perception function for the Gray-Scott reaction-diffusion system.
It computes the discrete Laplacian of each chemical species using a fixed convolution stencil,
which represents the spatial diffusion component of the PDE.
"""

from typing import Literal, override

import jax
import jax.numpy as jnp
from jax import Array

from cax.core.perceive import Perceive, grad2_kernel, identity_kernel


class ReactionDiffusionPerceive(Perceive[Array, Array]):
	"""Reaction-Diffusion perception.

	Computes the identity (current concentration) and the discrete Laplacian for each
	chemical species. The state has 2 channels (species U and V), so the perception
	produces 4 channels: [U, lap(U), V, lap(V)] via grouped convolution.

	The stencil is physics, not weights: it is stored as a plain constant, applied with
	`jax.lax.conv_general_dilated`, and never appears in the trainable parameters.
	"""

	def __init__(
		self,
		*,
		num_spatial_dims: int = 2,
		padding: Literal["CIRCULAR", "ZERO", "EDGE"] = "CIRCULAR",
	):
		"""Initialize Reaction-Diffusion perceive.

		Args:
			num_spatial_dims: Number of spatial dimensions (default 2).
			padding: Boundary condition mode. "CIRCULAR" for periodic boundaries, "ZERO"
				for an absorbing zero-concentration border, "EDGE" for a no-flux border.

		"""
		self.num_spatial_dims = num_spatial_dims
		self.pad_mode = {"CIRCULAR": "wrap", "ZERO": "constant", "EDGE": "edge"}[padding]
		self.channel_size = 2

		# One (identity, Laplacian) pair per species, shape (*kernel_spatial, 1, 4):
		# input size 1 per group and 4 output features under feature_group_count=2.
		kernel = jnp.concatenate(
			[
				identity_kernel(num_dims=num_spatial_dims),
				grad2_kernel(num_dims=num_spatial_dims, normalize=False),
			]
			* self.channel_size,
			axis=-1,
		)
		self.kernel = jnp.expand_dims(kernel, axis=-2)

	@override
	def __call__(self, state: Array) -> Array:
		"""Apply the fixed identity/Laplacian stencil to the input state.

		Args:
			state: Array with shape `(..., *spatial_dims, 2)` containing the [U, V]
				concentrations.

		Returns:
			Array with shape `(..., *spatial_dims, 4)` containing [U, lap(U), V, lap(V)].

		"""
		num_spatial_dims = self.num_spatial_dims
		spatial_dims = state.shape[-num_spatial_dims - 1 : -1]
		batch_dims = state.shape[: -num_spatial_dims - 1]

		# Pad the spatial axes per the boundary condition, then convolve without padding.
		pad_widths = [(0, 0)] * len(batch_dims) + [(1, 1)] * num_spatial_dims + [(0, 0)]
		padded = jnp.pad(state, pad_widths, mode=self.pad_mode)
		padded = padded.reshape(-1, *[dim + 2 for dim in spatial_dims], self.channel_size)

		# Channel-last layouts for any dimensionality: lhs (N, *spatial, C),
		# rhs (*spatial, I, O), out (N, *spatial, C).
		dimension_numbers = jax.lax.ConvDimensionNumbers(
			lhs_spec=(0, num_spatial_dims + 1, *range(1, num_spatial_dims + 1)),
			rhs_spec=(num_spatial_dims + 1, num_spatial_dims, *range(num_spatial_dims)),
			out_spec=(0, num_spatial_dims + 1, *range(1, num_spatial_dims + 1)),
		)
		perception = jax.lax.conv_general_dilated(
			padded,
			self.kernel,
			window_strides=(1,) * num_spatial_dims,
			padding="VALID",
			dimension_numbers=dimension_numbers,
			feature_group_count=self.channel_size,
		)

		return perception.reshape(*batch_dims, *spatial_dims, 2 * self.channel_size)
