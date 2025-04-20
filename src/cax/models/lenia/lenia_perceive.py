"""Lenia perceive module.

[1] Lenia - Biology of Artificial Life, Bert Wang-Chak Chan. 2019.
[2] Discovering Sensorimotor Agency in Cellular Automata using Diversity Search, Hamon, et al. 2024.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import Perceive
from cax.types import Perception, State

from .kernel import gaussian_kernel_fn
from .rule import RuleParams


class LeniaPerceive(Perceive):
	"""Lenia perception class."""

	def __init__(
		self,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		R: int,
		rule_params: RuleParams,
		*,
		state_scale: float,
		kernel_fn: Callable = gaussian_kernel_fn,
	):
		"""Initialize LeniaPerceive.

		Args:
			spatial_dims: Spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			rule_params: Parameters for the rules.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.

		"""
		super().__init__()
		self.spatial_dims = spatial_dims
		self.num_spatial_dims = len(spatial_dims)
		self.spatial_axes = tuple(range(self.num_spatial_dims))

		self.channel_size = channel_size
		self.R = R
		self.state_scale = state_scale
		self.kernel_fn = kernel_fn

		# Compute kernel fft
		self.kernel_fft = nnx.Param(self.compute_kernel_fft(rule_params))

		# Reshape channel to kernel
		self.reshape_channel_to_kernel = nnx.Param(
			self.compute_reshape_channel_to_kernel(rule_params)
		)

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
		state_fft_k = jnp.dot(state_fft, self.reshape_channel_to_kernel.value)

		# Compute kernel convolution
		U_k = jnp.real(jnp.fft.ifftn(self.kernel_fft * state_fft_k, axes=self.spatial_axes))

		return U_k

	@nnx.jit
	def update_rule_params(self, rule_params: RuleParams):
		"""Update the rule parameters."""
		# Compute kernel fft
		self.kernel_fft.value = self.compute_kernel_fft(rule_params)

		# Compute reshape channel to kernel
		self.reshape_channel_to_kernel.value = self.compute_reshape_channel_to_kernel(rule_params)

	@nnx.jit
	def compute_kernel_fft(self, rule_params: RuleParams) -> Array:
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

	@nnx.jit
	def compute_reshape_channel_to_kernel(self, rule_params: RuleParams) -> Array:
		"""Compute array to reshape from channel to kernel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size), out_axes=1)(
			rule_params.channel_source
		)
