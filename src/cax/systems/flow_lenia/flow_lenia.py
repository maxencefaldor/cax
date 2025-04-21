"""Flow Lenia model."""

from collections.abc import Callable

from flax import nnx
from jax import Array

from cax.core.ca import CA, metrics_fn
from cax.types import State
from cax.utils import clip_and_uint8, render_array_with_channels_to_rgb

from ..lenia.lenia_perceive import LeniaPerceive, gaussian_kernel_fn
from ..lenia.rule import RuleParams
from .flow_lenia_update import FlowLeniaUpdate, exponential_growth_fn


class FlowLenia(CA):
	"""Flow Lenia model."""

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
		# Flow Lenia parameters
		theta_A: float = 1.0,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize Lenia.

		Args:
			spatial_dims: Spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			T: Time resolution.
			rule_params: Parameters for the rules.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.
			growth_fn: Growth function.
			theta_A: Threshold for alpha in Flow Lenia.
			n: Exponent for alpha in Flow Lenia.
			dd: Maximum displacement distance.
			sigma: Spread parameter.
			metrics_fn: Metrics function.

		"""
		perceive = LeniaPerceive(
			spatial_dims=spatial_dims,
			channel_size=channel_size,
			R=R,
			rule_params=rule_params,
			state_scale=state_scale,
			kernel_fn=kernel_fn,
		)
		update = FlowLeniaUpdate(
			channel_size=channel_size,
			T=T,
			rule_params=rule_params,
			growth_fn=growth_fn,
			theta_A=theta_A,
			n=n,
			dd=dd,
			sigma=sigma,
		)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

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
