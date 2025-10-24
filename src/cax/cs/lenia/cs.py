"""Lenia module."""

from collections.abc import Callable

from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8, render_array_with_channels_to_rgb

from .metrics import metrics_fn
from .perceive import LeniaPerceive, gaussian_kernel_fn
from .rule import LeniaRuleParams
from .update import LeniaUpdate, exponential_growth_fn


class Lenia(ComplexSystem):
	"""Lenia class."""

	def __init__(
		self,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		*,
		R: int,
		T: int,
		state_scale: float = 1.0,
		kernel_fn: Callable = gaussian_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
		rule_params: LeniaRuleParams,
	):
		"""Initialize Lenia.

		Args:
			spatial_dims: Spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			T: Time resolution.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.
			growth_fn: Growth mapping function.
			rule_params: Parameters for the rules.

		"""
		self.perceive = LeniaPerceive(
			spatial_dims=spatial_dims,
			channel_size=channel_size,
			R=R,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
			rule_params=rule_params,
		)
		self.update = LeniaUpdate(
			channel_size=channel_size,
			T=T,
			growth_fn=growth_fn,
			rule_params=rule_params,
		)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			metrics = metrics_fn(next_state, R=self.perceive.R)
			self.sow(nnx.Intermediate, "state", next_state)
			self.sow(nnx.Intermediate, "metrics", metrics)

		return next_state

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = render_array_with_channels_to_rgb(state)

		return clip_and_uint8(rgb)
