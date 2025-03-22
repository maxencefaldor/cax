"""Lenia model."""

from collections.abc import Callable

from cax.core.ca import CA, metrics_fn
from flax import nnx

from .lenia_perceive import LeniaPerceive, gaussian_kernel_fn
from .lenia_update import LeniaUpdate, exponential_growth_fn
from .rule import RuleParams


class Lenia(CA):
	"""Lenia model."""

	def __init__(
		self,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		R: int,
		T: int,
		rule_params: RuleParams,
		*,
		state_scale: float = 1,
		kernel_fn: Callable = gaussian_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize Lenia."""
		perceive = LeniaPerceive(
			spatial_dims=spatial_dims,
			channel_size=channel_size,
			R=R,
			rule_params=rule_params,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
		)
		update = LeniaUpdate(
			channel_size=channel_size,
			T=T,
			rule_params=rule_params,
			growth_fn=growth_fn,
		)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@nnx.jit
	def update_rule_params(self, rule_params: RuleParams):
		"""Update the rule parameters."""
		self.perceive.update_rule_params(rule_params)
		self.update.update_rule_params(rule_params)
