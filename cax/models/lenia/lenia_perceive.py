"""Lenia perceive module.

[1] Lenia - Biology of Artificial Life, Bert Wang-Chak Chan. 2019.
[2] Discovering Sensorimotor Agency in Cellular Automata using Diversity Search, Hamon, et al. 2024.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State
from flax import nnx
from jax import Array

from .types import FreeKernelParams, OriginalKernelParams, RuleParams


def bell(x: Array, mean: Array, std: Array) -> Array:
	"""Gaussian function."""
	return jnp.exp(-0.5 * ((x - mean) / std) ** 2)


def original_kernel_fn(radius: Array, kernel_params: OriginalKernelParams) -> Array:
	"""Original kernel function introduced in [1]."""
	# Compute kernel mask to avoid out of bounds interactions
	mask = radius < kernel_params.r

	# Compute segment index and position in segment
	number_of_segments = jnp.count_nonzero(~jnp.isnan(kernel_params.b), axis=-1)

	segment_position = radius * number_of_segments / kernel_params.r
	segment_index = jnp.minimum(segment_position.astype(int), number_of_segments - 1)
	position_in_segment = segment_position % 1

	return mask * kernel_params.b[segment_index] * bell(position_in_segment, 0.5, 0.15)


def free_kernel_fn(radius: Array, kernel_params: FreeKernelParams) -> Array:
	"""Free kernel function introduced in [2]."""
	# Compute soft kernel mask to avoid out of bounds interactions
	# mask = radius < kernel_params.r
	mask = nnx.sigmoid(-10 * (radius - 1))

	return mask * jnp.sum(
		kernel_params.b
		* jax.vmap(bell, in_axes=(None, 0, 0), out_axes=-1)(
			radius / kernel_params.r, kernel_params.a, kernel_params.w
		),
		axis=-1,
	)


class LeniaPerceive(Perceive):
	"""Lenia perception class."""

	kernel_fft: nnx.Param
	reshape_channel_to_kernel: nnx.Param

	def __init__(
		self,
		num_dims: int,
		channel_size: int,
		R: int,
		rule_params: RuleParams,
		state_size: int,
		state_scale: float,
		kernel_fn: Callable = original_kernel_fn,
	):
		"""Initialize LeniaPerceive.

		Args:
			num_dims: Number of spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			rule_params: Parameters for the rules.
			state_size: State size.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.

		"""
		super().__init__()
		self.num_dims = num_dims
		self.channel_size = channel_size
		self.R = R
		self.state_size = state_size
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
		state_fft = jnp.fft.fft2(state, axes=(-3, -2))  # (y, x, c,)

		# Deaggregate channels for kernel convolution
		state_fft_k = jnp.dot(state_fft, self.reshape_channel_to_kernel.value)  # (y, x, k,)

		# Compute kernel convolution
		U_k = jnp.real(jnp.fft.ifft2(self.kernel_fft * state_fft_k, axes=(-3, -2)))  # (y, x, k,)

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
		mid = self.state_size // 2
		x = jnp.mgrid[*[slice(-mid, mid) for _ in range(self.num_dims)]] / (
			self.state_scale * self.R
		)
		d = jnp.linalg.norm(x, axis=0)

		# Compute kernel
		kernel = nnx.vmap(self.kernel_fn, in_axes=(None, 0), out_axes=-1)(
			d, rule_params.kernel_params
		)

		# Normalize kernel
		kernel_normalized = kernel / jnp.sum(kernel, axis=(0, 1), keepdims=True)  # (y, x, k,)

		# Compute kernel fft
		kernel_fft = jnp.fft.fft2(
			jnp.fft.fftshift(kernel_normalized, axes=(0, 1)), axes=(0, 1)
		)  # (y, x, k,)

		return kernel_fft

	@nnx.jit
	def compute_reshape_channel_to_kernel(self, rule_params: RuleParams) -> Array:
		"""Compute array to reshape from channel to kernel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size), out_axes=1)(
			rule_params.channel_source
		)
