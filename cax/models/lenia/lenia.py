"""Lenia model."""

from collections.abc import Callable

from cax.core.ca import CA

from .lenia_perceive import LeniaPerceive, original_kernel_fn
from .lenia_update import LeniaUpdate, growth_exponential
from .types import RuleParams


class Lenia(CA):
	"""Lenia model."""

	def __init__(
		self,
		num_dims: int,
		state_size: int,
		R: int,
		T: int,
		rules_params: RuleParams,
		state_scale: float = 1,
		kernel_fn: Callable = original_kernel_fn,
		growth_fn: Callable = growth_exponential,
		flow: bool = False,
	):
		"""Initialize Lenia."""
		perceive = LeniaPerceive(num_dims, state_size, R, state_scale, rules_params, kernel_fn)
		update = LeniaUpdate(T, state_scale, rules_params, growth_fn, flow)

		super().__init__(perceive, update)
