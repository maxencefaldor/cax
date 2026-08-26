"""Particle Lenia module.

References:
	[1] Particle Lenia and the energy-based formulation, Mordvintsev et al. 2022.
		https://google-research.github.io/self-organising-systems/particle-lenia/

"""

from .cs import ParticleLenia
from .growth import ParticleLeniaGrowthParams, peak_growth_fn
from .kernel import ParticleLeniaKernelParams, bell, peak_kernel_fn
from .perceive import ParticleLeniaPerceive
from .rule import ParticleLeniaRuleParams
from .update import ParticleLeniaUpdate

__all__ = [
	"ParticleLenia",
	"ParticleLeniaGrowthParams",
	"ParticleLeniaKernelParams",
	"ParticleLeniaPerceive",
	"ParticleLeniaRuleParams",
	"ParticleLeniaUpdate",
	"bell",
	"peak_growth_fn",
	"peak_kernel_fn",
]
