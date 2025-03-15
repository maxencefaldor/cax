"""Particle Lenia types."""

from flax import struct
from jax import Array


@struct.dataclass
class KernelParams:
	weight: Array
	mean: Array
	std: Array


@struct.dataclass
class GrowthParams:
	mean: Array
	std: Array


@struct.dataclass
class RuleParams:
	c_rep: Array
	kernel_params: KernelParams
	growth_params: GrowthParams
