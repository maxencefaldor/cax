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
		channel_size: int,
		R: int,
		T: int,
		rules_params: RuleParams,
		state_size: int,
		state_scale: float = 1,
		kernel_fn: Callable = original_kernel_fn,
		growth_fn: Callable = growth_exponential,
	):
		"""Initialize Lenia."""
		perceive = LeniaPerceive(
			num_dims=num_dims,
			channel_size=channel_size,
			R=R,
			rules_params=rules_params,
			state_size=state_size,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
		)
		update = LeniaUpdate(
			channel_size=channel_size,
			T=T,
			rules_params=rules_params,
			growth_fn=growth_fn,
		)

		super().__init__(perceive, update)
