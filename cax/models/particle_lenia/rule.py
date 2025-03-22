"""Particle Lenia rule."""

from flax import struct
from jax import Array

from .growth import GrowthParams
from .kernel import KernelParams


@struct.dataclass
class RuleParams:
	"""Rule parameters."""

	c_rep: Array
	kernel_params: KernelParams
	growth_params: GrowthParams
