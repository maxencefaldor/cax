"""Lenia perceive module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import Perceive
from cax.types import Perception, State

from .kernel import gaussian_kernel_fn
from .rule import LeniaRuleParams


class LeniaPerceive(Perceive):
	"""Lenia perception class."""

	def __init__(
		self,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		*,
		R: int,
		state_scale: float = 1.0,
		kernel_fn: Callable = gaussian_kernel_fn,
		rule_params: LeniaRuleParams,
	):
		"""Initialize Lenia perceive.

		Args:
			spatial_dims: Spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.
			rule_params: Parameters for the rules.

		"""
		self.spatial_dims = spatial_dims
		self.num_spatial_dims = len(spatial_dims)
		self.spatial_axes = tuple(range(self.num_spatial_dims))
		self.channel_size = channel_size
		self.R = R
		self.state_scale = state_scale

		self.reshape_channel_to_kernel = self._reshape_channel_to_kernel(rule_params)

		self.kernel_fn = kernel_fn
		self.kernel_fft = self._kernel_fft(rule_params)

	def __call__(self, state: State) -> Perception:
		"""Apply Lenia perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying Lenia convolution.

		"""
		# Compute state fft
		state_fft = jnp.fft.fftn(state, axes=self.spatial_axes)

		# Deaggregate channels for kernel convolution
		state_fft_k = jnp.dot(state_fft, self.reshape_channel_to_kernel)

		# Compute kernel convolution
		U_k = jnp.real(jnp.fft.ifftn(self.kernel_fft * state_fft_k, axes=self.spatial_axes))

		return U_k

	def _kernel_fft(self, rule_params: LeniaRuleParams) -> Array:
		"""Compute the kernel fft based on the kernel function and rules parameters."""
		x = jnp.mgrid[[slice(-dim // 2, dim // 2) for dim in self.spatial_dims]] / (
			self.state_scale * self.R
		)
		d = jnp.linalg.norm(x, axis=0)

		# Compute kernel
		kernel = nnx.vmap(self.kernel_fn, in_axes=(None, 0), out_axes=-1)(
			d, rule_params.kernel_params
		)

		# Normalize kernel
		kernel_normalized = kernel / jnp.sum(kernel, axis=self.spatial_axes, keepdims=True)

		# Compute kernel fft
		kernel_fft = jnp.fft.fftn(
			jnp.fft.fftshift(kernel_normalized, axes=self.spatial_axes), axes=self.spatial_axes
		)

		return kernel_fft

	def _reshape_channel_to_kernel(self, rule_params: LeniaRuleParams) -> Array:
		"""Compute array to reshape from channel to kernel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size), out_axes=1)(
			rule_params.channel_source
		)
