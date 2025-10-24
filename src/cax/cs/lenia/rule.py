"""Lenia rule parameters module."""

from flax import struct

from .growth import GrowthParams
from .kernel import KernelParams


@struct.dataclass
class LeniaRuleParams:
	"""Lenia rule parameters class."""

	channel_source: int
	channel_target: int
	weight: float
	kernel_params: KernelParams
	growth_params: GrowthParams
