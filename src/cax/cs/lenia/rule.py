"""Lenia rule parameters module."""

from flax import nnx
from jax import Array

from .growth import GrowthParams
from .kernel import KernelParams


@nnx.dataclass
class LeniaRuleParams(nnx.Pytree):
	"""Lenia rule parameters class."""

	channel_source: Array = nnx.data()
	channel_target: Array = nnx.data()
	weight: Array = nnx.data()
	kernel_params: KernelParams = nnx.data()
	growth_params: GrowthParams = nnx.data()
