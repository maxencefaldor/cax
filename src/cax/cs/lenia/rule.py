"""Lenia rule parameters module."""

from flax import nnx
from jax import Array

from .growth import GrowthParams
from .kernel import KernelParams


@nnx.dataclass
class LeniaRuleParams(nnx.Pytree):
	"""Lenia rule parameters class."""

	channel_source: Array
	channel_target: Array
	weight: Array
	kernel_params: KernelParams
	growth_params: GrowthParams
