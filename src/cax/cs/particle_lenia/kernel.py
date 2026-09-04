"""Particle Lenia kernel module.

References:
    [1] Particle Lenia and the energy-based formulation, Mordvintsev et al. 2022.
        https://google-research.github.io/self-organising-systems/particle-lenia/

"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParticleLeniaKernelParams:
    """Kernel parameters.

    Attributes:
        weight: Kernel weight.
        mean: Center of the kernel bump.
        std: Width of the kernel bump.

    """

    weight: Array
    mean: Array
    std: Array


def bell(x: Array, mean: Array, std: Array) -> Array:
    """Gaussian bump, `exp(-((x - mean) / std)^2)` — Particle Lenia's convention [1].

    Grid Lenia's `bell` deliberately differs (1/2 factor in the exponent); each mirrors
    its reference, and parameters do not transfer between the two without rescaling.
    """
    return jnp.exp(-(((x - mean) / std) ** 2))


def peak_kernel_fn(radius: Array, kernel_params: ParticleLeniaKernelParams) -> Array:
    """Peak kernel function introduced in [1]."""
    return kernel_params.weight * bell(radius, kernel_params.mean, kernel_params.std)
