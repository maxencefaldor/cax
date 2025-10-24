"""Particle Lenia rule parameters module."""

from flax import struct

from .growth import GrowthParams
from .kernel import KernelParams


@struct.dataclass
class ParticleLeniaRuleParams:
	"""Particle Lenia rule parameters class."""

	c_rep: float
	kernel_params: KernelParams
	growth_params: GrowthParams
