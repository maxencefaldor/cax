"""Particle Lenia rule parameters module."""

from flax import nnx

from .growth import GrowthParams
from .kernel import KernelParams


class ParticleLeniaRuleParams(nnx.Pytree):
	"""Particle Lenia rule parameters class."""

	def __init__(self, c_rep: float, kernel_params: KernelParams, growth_params: GrowthParams):
		"""Initialize Particle Lenia rule parameters."""
		self.c_rep = c_rep
		self.kernel_params = kernel_params
		self.growth_params = growth_params
