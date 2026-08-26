"""Lenia module.

This module implements Lenia, a continuous cellular automaton that generalizes Conway's
Game of Life to continuous space, time, and states. Lenia features smooth state transitions,
continuous kernels, and growth functions that enable the emergence of diverse life-like
patterns and behaviors. The system supports multiple channels and arbitrary spatial dimensions.
"""

from collections.abc import Callable

from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8, render_array_with_channels_to_rgb

from .growth import exponential_growth_fn
from .kernel import gaussian_kernel_fn
from .perceive import LeniaPerceive
from .rule import LeniaRuleParams
from .update import LeniaUpdate


class Lenia(ComplexSystem[Array, Array]):
	"""Lenia class."""

	def __init__(
		self,
		spatial_dims: tuple[int, ...],
		channel_size: int,
		*,
		R: int,
		T: float,
		state_scale: float = 1.0,
		kernel_fn: Callable = gaussian_kernel_fn,
		growth_fn: Callable = exponential_growth_fn,
		rule_params: LeniaRuleParams,
	):
		"""Initialize Lenia.

		Args:
			spatial_dims: Spatial dimensions (e.g., (64, 64) for 2D or (32, 32, 32) for 3D).
			channel_size: Number of channels.
			R: Space resolution defining the kernel radius. Larger values create wider
				neighborhoods and smoother patterns.
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.
			state_scale: Scaling factor applied to state values.
			kernel_fn: Callable that generates convolution kernels. Takes rule parameters
				and returns kernel weights.
			growth_fn: Callable that maps neighborhood potential to growth values. Defines
				how cells respond to their local environment.
			rule_params: Instance of LeniaRuleParams containing kernel and growth parameters
				for each channel.

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

	def _step(self, state: Array, input: Array | None = None) -> Array:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		return next_state

	@nnx.jit
	def render(self, state: Array) -> Array:
		"""Render state to RGB image.

		Converts the multi-channel Lenia state to an RGB visualization. Channels are
		mapped to color channels (Red, Green, Blue) for visualization. If there are
		more than 3 channels, the last three are displayed. If there are fewer than
		3 channels, the missing channels are filled with zeros.

		Args:
			state: Array with shape (*spatial_dims, channel_size) representing the
				Lenia state, where each cell contains continuous values typically in [0, 1].

		Returns:
			RGB image with dtype uint8 and shape (*spatial_dims, 3), where state
				values are mapped to colors in the range [0, 255].

		"""
		rgb = render_array_with_channels_to_rgb(state)

		return clip_and_uint8(rgb)
