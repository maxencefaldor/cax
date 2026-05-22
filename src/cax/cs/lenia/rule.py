"""Lenia rule parameters module."""

from flax import nnx
from jax import Array

from .growth import GrowthParams
from .kernel import KernelParams


class LeniaRuleParams(nnx.Pytree):
	"""Lenia rule parameters class."""

	def __init__(
		self,
		channel_source: Array,
		channel_target: Array,
		weight: Array,
		kernel_params: KernelParams,
		growth_params: GrowthParams,
	):
		"""Initialize Lenia rule parameters."""
		self.channel_source = channel_source
		self.channel_target = channel_target
		self.weight = weight
		self.kernel_params = kernel_params
		self.growth_params = growth_params
