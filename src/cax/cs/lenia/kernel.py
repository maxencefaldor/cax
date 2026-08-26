"""Lenia kernel module.

References:
	[1] Lenia — Biology of Artificial Life, Bert Wang-Chak Chan. 2019.
	[2] Discovering Sensorimotor Agency in Cellular Automata using Diversity Search,
		Hamon et al. 2024.

"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


@nnx.dataclass
class LeniaKernelParams(nnx.Pytree):
	"""Kernel parameters."""

	r: Array = nnx.data()
	beta: Array = nnx.data()


@nnx.dataclass
class FreeKernelParams(nnx.Pytree):
	"""Free kernel parameters from [2]."""

	r: Array = nnx.data()
	b: Array = nnx.data()
	a: Array = nnx.data()
	w: Array = nnx.data()


def bell(x: Array, mean: Array | float, std: Array | float) -> Array:
	"""Gaussian function, `exp(-((x - mean) / std)^2 / 2)` — grid Lenia's convention [1].

	Particle Lenia's `bell` deliberately differs (no 1/2 factor); each mirrors its
	reference, and parameters do not transfer between the two without rescaling.
	"""
	return jnp.exp(-0.5 * ((x - mean) / std) ** 2)


def get_kernel_fn(
	kernel_core: Callable[[Array], Array],
) -> Callable[[Array, LeniaKernelParams], Array]:
	"""Get kernel function."""

	def kernel_fn(radius: Array, kernel_params: LeniaKernelParams) -> Array:
		"""Kernel function."""
		mask = radius < kernel_params.r

		# Compute segment index and position in segment
		rank = jnp.count_nonzero(~jnp.isnan(kernel_params.beta), axis=-1)

		segment_position = radius * rank / kernel_params.r
		segment_idx = jnp.minimum(segment_position.astype(int), rank - 1)
		position_in_segment = segment_position % 1

		return mask * kernel_params.beta[segment_idx] * kernel_core(position_in_segment)

	return kernel_fn


# Kernel cores
def exponential_kernel_core(radius: Array, alpha: float = 4.0) -> Array:
	"""Exponential kernel core.

	The core is zero at the support boundaries, where the exponent diverges. The
	double-`where` sanitizes the divisor before dividing so the gradient stays finite
	at `radius` 0 and 1 (see cax.utils.numerics).
	"""
	is_interior = (radius > 0.0) & (radius < 1.0)
	support = jnp.where(is_interior, 4 * radius * (1 - radius), jnp.ones_like(radius))
	return jnp.where(is_interior, jnp.exp(alpha - alpha / support), jnp.zeros_like(radius))


def polynomial_kernel_core(radius: Array, alpha: float = 4.0) -> Array:
	"""Polynomial kernel core."""
	return (4 * radius * (1 - radius)) ** alpha


def rectangular_kernel_core(radius: Array) -> Array:
	"""Rectangular kernel core."""
	return jnp.where((radius >= 1 / 4) & (radius <= 3 / 4), 1.0, 0.0)


def gaussian_kernel_core(radius: Array, std: float = 0.15) -> Array:
	"""Gaussian kernel core."""
	return bell(radius, 0.5, std)


# Kernel shells
exponential_kernel_fn = get_kernel_fn(exponential_kernel_core)
polynomial_kernel_fn = get_kernel_fn(polynomial_kernel_core)
rectangular_kernel_fn = get_kernel_fn(rectangular_kernel_core)
gaussian_kernel_fn = get_kernel_fn(gaussian_kernel_core)


# Differentiable kernel
def free_kernel_fn(radius: Array, kernel_params: FreeKernelParams) -> Array:
	"""Free kernel function introduced in [2].

	Follows [2]'s convention exactly: Gaussian bumps `exp(-((x/r - a) / w)^2 / 2)` under a
	sigmoid support mask. The official Flow Lenia repository uses a different bump
	convention (variance-form widths, support scaled with r), so kernel parameters from
	that codebase do not transfer numerically to this function.
	"""
	# Compute soft kernel mask to avoid out of bounds interactions
	mask = nnx.sigmoid(-10 * (radius - 1))

	return mask * jnp.sum(
		kernel_params.b
		* jax.vmap(bell, in_axes=(None, 0, 0), out_axes=-1)(
			radius / kernel_params.r, kernel_params.a, kernel_params.w
		),
		axis=-1,
	)
