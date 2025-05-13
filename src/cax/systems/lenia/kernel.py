"""Lenia kernel module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import Array


@struct.dataclass
class KernelParams:
	"""Kernel parameters."""

	r: Array
	b: Array


@struct.dataclass
class FreeKernelParams:
	"""Free kernel parameters from https://arxiv.org/abs/2212.07906."""

	r: Array
	b: Array
	a: Array
	w: Array


def bell(x: Array, mean: Array, std: Array) -> Array:
	"""Gaussian function."""
	return jnp.exp(-0.5 * ((x - mean) / std) ** 2)


def get_kernel_fn(kernel_core: Callable) -> Callable:
	"""Get kernel function."""

	def kernel_fn(radius: Array, kernel_params: KernelParams) -> Array:
		"""Kernel function."""
		mask = radius < kernel_params.r

		# Compute segment index and position in segment
		number_of_segments = jnp.count_nonzero(~jnp.isnan(kernel_params.b), axis=-1)

		segment_position = radius * number_of_segments / kernel_params.r
		segment_idx = jnp.minimum(segment_position.astype(int), number_of_segments - 1)
		position_in_segment = segment_position % 1

		return mask * kernel_params.b[segment_idx] * kernel_core(position_in_segment)

	return kernel_fn


# Kernel cores
def exponential_kernel_core(radius, alpha=4):
	"""Exponential kernel core."""
	return jnp.exp(alpha - alpha / (4 * radius * (1 - radius)))


def polynomial_kernel_core(radius, alpha=4):
	"""Polynomial kernel core."""
	return (4 * radius * (1 - radius)) ** alpha


def rectangular_kernel_core(radius):
	"""Rectangular kernel core."""
	return jnp.where((radius >= 1 / 4) & (radius <= 3 / 4), 1, 0)


def gaussian_kernel_core(radius):
	"""Gaussian kernel core."""
	return bell(radius, 0.5, 0.15)


# Kernel shells
exponential_kernel_fn = get_kernel_fn(exponential_kernel_core)
polynomial_kernel_fn = get_kernel_fn(polynomial_kernel_core)
rectangular_kernel_fn = get_kernel_fn(rectangular_kernel_core)
gaussian_kernel_fn = get_kernel_fn(gaussian_kernel_core)


# Differentiable kernel
def free_kernel_fn(radius: Array, kernel_params: FreeKernelParams) -> Array:
	"""Free kernel function introduced in [2]."""
	# Compute soft kernel mask to avoid out of bounds interactions
	mask = nnx.sigmoid(-10 * (radius - 1))

	return mask * jnp.sum(
		kernel_params.b
		* jax.vmap(bell, in_axes=(None, 0, 0), out_axes=-1)(
			radius / kernel_params.r, kernel_params.a, kernel_params.w
		),
		axis=-1,
	)
