"""Lenia module.

[1] Lenia - Biology of Artificial Life, Bert Wang-Chak Chan. 2019.
[2] Discovering Sensorimotor Agency in Cellular Automata using Diversity Search, Hamon, et al. 2024.
"""

from .cs import Lenia
from .growth import GrowthParams, exponential_growth_fn
from .kernel import (
	FreeKernelParams,
	KernelParams,
	exponential_kernel_fn,
	free_kernel_fn,
	gaussian_kernel_fn,
	polynomial_kernel_fn,
	rectangular_kernel_fn,
)
from .perceive import LeniaPerceive
from .rule import LeniaRuleParams
from .update import LeniaUpdate

__all__ = [
	"Lenia",
	"LeniaPerceive",
	"LeniaUpdate",
	"KernelParams",
	"FreeKernelParams",
	"exponential_kernel_fn",
	"gaussian_kernel_fn",
	"polynomial_kernel_fn",
	"rectangular_kernel_fn",
	"free_kernel_fn",
	"GrowthParams",
	"exponential_growth_fn",
	"LeniaRuleParams",
]
