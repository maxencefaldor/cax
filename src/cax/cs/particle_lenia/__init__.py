"""Particle Lenia module.

[1] https://google-research.github.io/self-organising-systems/particle-lenia/
"""

from ..lenia.growth import exponential_growth_fn
from .cs import ParticleLenia
from .growth import GrowthParams
from .kernel import KernelParams, bell, peak_kernel_fn
from .perceive import ParticleLeniaPerceive
from .rule import ParticleLeniaRuleParams
from .update import ParticleLeniaUpdate

__all__ = [
	"ParticleLenia",
	"ParticleLeniaPerceive",
	"ParticleLeniaUpdate",
	"KernelParams",
	"bell",
	"peak_kernel_fn",
	"exponential_growth_fn",
	"GrowthParams",
	"ParticleLeniaRuleParams",
]
