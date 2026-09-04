"""Lenia rule parameters module."""

from dataclasses import dataclass

import jax
from jax import Array

from .growth import LeniaGrowthParams
from .kernel import LeniaKernelParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LeniaRuleParams:
    """Lenia rule parameters class.

    Attributes:
        channel_source: Channel each kernel reads from.
        channel_target: Channel each kernel's growth is applied to.
        weight: Kernel mixing weights, normalized by their sum before use.
        kernel_params: Kernel parameters, one entry per kernel.
        growth_params: Growth parameters, one entry per kernel.

    """

    channel_source: Array
    channel_target: Array
    weight: Array
    kernel_params: LeniaKernelParams
    growth_params: LeniaGrowthParams
