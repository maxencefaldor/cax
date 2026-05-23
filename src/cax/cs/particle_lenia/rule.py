"""Particle Lenia rule parameters module."""

from flax import nnx

from .growth import GrowthParams
from .kernel import KernelParams


@nnx.dataclass
class ParticleLeniaRuleParams(nnx.Pytree):
	"""Particle Lenia rule parameters class."""

	c_rep: float = nnx.static()
	kernel_params: KernelParams
	growth_params: GrowthParams
