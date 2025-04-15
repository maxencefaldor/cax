"""Particle Lenia kernel module.

[1] https://google-research.github.io/self-organising-systems/particle-lenia/
"""

import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class KernelParams:
	"""Kernel parameters."""

	weight: Array
	mean: Array
	std: Array


def bell(x: Array, mean: Array, std: Array) -> Array:
	"""Gaussian function."""
	return jnp.exp(-(((x - mean) / std) ** 2))


def peak_kernel_fn(radius: Array, kernel_params: KernelParams) -> Array:
	"""Peak kernel function introduced in [1]."""
	return kernel_params.weight * bell(radius, kernel_params.mean, kernel_params.std)
