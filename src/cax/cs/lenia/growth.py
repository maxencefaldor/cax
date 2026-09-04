"""Lenia growth module.

References:
    [1] Lenia — Biology of Artificial Life, Bert Wang-Chak Chan. 2019.

"""

from dataclasses import dataclass

import jax
from jax import Array

from .kernel import bell


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LeniaGrowthParams:
    """Growth parameters.

    Attributes:
        mean: Center of the growth bell.
        std: Width of the growth bell.

    """

    mean: Array
    std: Array


def exponential_growth_fn(u: Array, growth_params: LeniaGrowthParams) -> Array:
    """Growth mapping function introduced in [1]."""
    return 2 * bell(u, growth_params.mean, growth_params.std) - 1
