"""Particle Lenia rule parameters module."""

from flax import nnx

from .growth import ParticleLeniaGrowthParams
from .kernel import ParticleLeniaKernelParams


@nnx.dataclass
class ParticleLeniaRuleParams(nnx.Pytree):
	"""Particle Lenia rule parameters class."""

	c_rep: float = nnx.static()
	kernel_params: ParticleLeniaKernelParams = nnx.data()
	growth_params: ParticleLeniaGrowthParams = nnx.data()
