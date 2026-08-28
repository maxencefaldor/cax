"""Particle Lenia rule parameters module."""

from dataclasses import dataclass, field

import jax

from .growth import ParticleLeniaGrowthParams
from .kernel import ParticleLeniaKernelParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ParticleLeniaRuleParams:
	"""Particle Lenia rule parameters class.

	Attributes:
		c_rep: Repulsion strength. Static: part of the structure, not a leaf.
		kernel_params: Kernel parameters.
		growth_params: Growth parameters.

	"""

	c_rep: float = field(metadata={"static": True})
	kernel_params: ParticleLeniaKernelParams
	growth_params: ParticleLeniaGrowthParams
