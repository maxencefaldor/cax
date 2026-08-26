"""Lenia module.

References:
	[1] Lenia — Biology of Artificial Life, Bert Wang-Chak Chan. 2019.
	[2] Discovering Sensorimotor Agency in Cellular Automata using Diversity Search,
		Hamon et al. 2024.

"""

from .cs import Lenia
from .growth import LeniaGrowthParams, exponential_growth_fn
from .kernel import (
	FreeKernelParams,
	LeniaKernelParams,
	exponential_kernel_fn,
	free_kernel_fn,
	gaussian_kernel_fn,
	polynomial_kernel_fn,
	rectangular_kernel_fn,
)
from .metrics import center_state, metrics_fn
from .patterns import PATTERN_NAMES, load_pattern
from .perceive import LeniaPerceive
from .rule import LeniaRuleParams
from .update import LeniaUpdate

__all__ = [
	"PATTERN_NAMES",
	"FreeKernelParams",
	"LeniaGrowthParams",
	"LeniaKernelParams",
	"Lenia",
	"LeniaPerceive",
	"LeniaRuleParams",
	"LeniaUpdate",
	"center_state",
	"exponential_growth_fn",
	"exponential_kernel_fn",
	"free_kernel_fn",
	"gaussian_kernel_fn",
	"load_pattern",
	"metrics_fn",
	"polynomial_kernel_fn",
	"rectangular_kernel_fn",
]
