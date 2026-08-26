"""Flow Lenia module.

Flow Lenia shares Lenia's perception: the FFT convolution that computes potential
fields is identical, so `LeniaPerceive` is re-exported here under its own name —
what differs is the update, which turns growth into mass-conserving flow.

References:
	[1] Flow-Lenia: Towards open-ended evolution in cellular automata through mass
		conservation and parameter localization, Plantec et al. 2023.

"""

from ..lenia.growth import LeniaGrowthParams, exponential_growth_fn
from ..lenia.kernel import (
	FreeKernelParams,
	LeniaKernelParams,
	exponential_kernel_fn,
	free_kernel_fn,
	gaussian_kernel_fn,
	polynomial_kernel_fn,
	rectangular_kernel_fn,
)
from ..lenia.perceive import LeniaPerceive
from ..lenia.rule import LeniaRuleParams
from .cs import FlowLenia
from .update import FlowLeniaUpdate

__all__ = [
	"FlowLenia",
	"FlowLeniaUpdate",
	"FreeKernelParams",
	"LeniaGrowthParams",
	"LeniaKernelParams",
	"LeniaPerceive",
	"LeniaRuleParams",
	"exponential_growth_fn",
	"exponential_kernel_fn",
	"free_kernel_fn",
	"gaussian_kernel_fn",
	"polynomial_kernel_fn",
	"rectangular_kernel_fn",
]
