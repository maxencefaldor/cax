"""Lenia types."""

from flax import struct
from jax import Array


@struct.dataclass
class KernelParams:
	pass


@struct.dataclass
class OriginalKernelParams(KernelParams):
	r: Array
	b: Array


@struct.dataclass
class FreeKernelParams(KernelParams):
	r: Array
	b: Array
	a: Array
	w: Array


@struct.dataclass
class GrowthParams:
	mean: Array
	std: Array


@struct.dataclass
class RuleParams:
	channel_source: int
	channel_target: int
	weight: float
	kernel_params: KernelParams
	growth_params: GrowthParams
