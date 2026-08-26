"""Particle Lenia kernel module.

References:
	[1] Particle Lenia and the energy-based formulation, Mordvintsev et al. 2022.
		https://google-research.github.io/self-organising-systems/particle-lenia/

"""

import jax.numpy as jnp
from flax import nnx
from jax import Array


@nnx.dataclass
class KernelParams(nnx.Pytree):
	"""Kernel parameters."""

	weight: Array = nnx.data()
	mean: Array = nnx.data()
	std: Array = nnx.data()


def bell(x: Array, mean: Array, std: Array) -> Array:
	"""Gaussian bump, `exp(-((x - mean) / std)^2)` — Particle Lenia's convention [1].

	Grid Lenia's `bell` deliberately differs (1/2 factor in the exponent); each mirrors
	its reference, and parameters do not transfer between the two without rescaling.
	"""
	return jnp.exp(-(((x - mean) / std) ** 2))


def peak_kernel_fn(radius: Array, kernel_params: KernelParams) -> Array:
	"""Peak kernel function introduced in [1]."""
	return kernel_params.weight * bell(radius, kernel_params.mean, kernel_params.std)
