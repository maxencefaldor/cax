"""Lenia model."""

from collections.abc import Callable

from flax import nnx
from jax import Array

from cax.core.ca import CA, metrics_fn
from cax.types import State
from cax.utils import clip_and_uint8, render_array_with_channels_to_rgb

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

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = render_array_with_channels_to_rgb(state)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
