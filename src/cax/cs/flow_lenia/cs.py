"""Flow Lenia module."""

from collections.abc import Callable

from flax import nnx
from jax import Array

from cax.core.cs import ComplexSystem
from cax.types import Input, State
from cax.utils import clip_and_uint8, render_array_with_channels_to_rgb

from ..lenia.perceive import LeniaPerceive, gaussian_kernel_fn
from ..lenia.rule import LeniaRuleParams
from .update import FlowLeniaUpdate, exponential_growth_fn


class FlowLenia(ComplexSystem):
	"""Flow Lenia class."""

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
		# Flow Lenia parameters
		theta_A: float = 1.0,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
	):
		"""Initialize Flow Lenia.

		Args:
			spatial_dims: Spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			T: Time resolution.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.
			growth_fn: Growth mapping function.
			rule_params: Parameters for the rules.
			theta_A: Threshold for alpha in Flow Lenia.
			n: Exponent for alpha in Flow Lenia.
			dd: Maximum displacement distance.
			sigma: Spread parameter.

		"""
		self.perceive = LeniaPerceive(
			spatial_dims=spatial_dims,
			channel_size=channel_size,
			R=R,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
			rule_params=rule_params,
		)
		self.update = FlowLeniaUpdate(
			channel_size=channel_size,
			T=T,
			growth_fn=growth_fn,
			rule_params=rule_params,
			theta_A=theta_A,
			n=n,
			dd=dd,
			sigma=sigma,
		)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

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
