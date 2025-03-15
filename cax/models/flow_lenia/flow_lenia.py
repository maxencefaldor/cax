"""Flow Lenia model."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from cax.core.ca import CA
from cax.types import State
from cax.utils.render import clip_and_uint8
from flax import nnx

from ..lenia.lenia_perceive import LeniaPerceive, original_kernel_fn
from ..lenia.types import RuleParams
from .flow_lenia_update import FlowLeniaUpdate, growth_exponential


class FlowLenia(CA):
	"""Flow Lenia model."""

	def __init__(
		self,
		num_dims: int,
		channel_size: int,
		R: int,
		T: int,
		rule_params: RuleParams,
		state_size: int,
		state_scale: float = 1,
		kernel_fn: Callable = original_kernel_fn,
		growth_fn: Callable = growth_exponential,
		theta_A: float = 1.0,
		n: int = 2,
		dd: int = 5,
		sigma: float = 0.65,
		boundary: str = "CIRCULAR",
	):
		"""Initialize Lenia.

		Args:
			num_dims: Number of spatial dimensions.
			channel_size: Number of channels.
			R: Space resolution.
			T: Time resolution.
			rule_params: Parameters for the rules.
			state_size: Size of each spatial dimension.
			state_scale: Scaling factor for the state.
			kernel_fn: Kernel function.
			growth_fn: Growth function.
			theta_A: Threshold for alpha in Flow Lenia.
			n: Exponent for alpha in Flow Lenia.
			dd: Maximum displacement distance.
			sigma: Spread parameter.
			boundary: Boundary condition.

		"""
		self.num_dims = num_dims
		perceive = LeniaPerceive(
			num_dims=num_dims,
			channel_size=channel_size,
			R=R,
			rule_params=rule_params,
			state_size=state_size,
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
			boundary=boundary,
		)

		super().__init__(perceive, update)

	@nnx.jit
	def render(self, state: State) -> jax.Array:
		"""Render state.

		Args:
			state: An array of states.

		Returns:
			Rendered states.

		"""
		assert self.num_dims == 2, "Flow Lenia only supports 2D visualization."

		if state.shape[-1] == 1:
			frame = jnp.repeat(state, 3, axis=-1)
		elif state.shape[-1] == 2:
			red = state[..., 0:1]
			green = state[..., 1:2]
			blue = jnp.zeros_like(red)
			frame = jnp.concatenate([red, green, blue], axis=-1)
		else:
			frame = state[..., :3]

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(frame)
