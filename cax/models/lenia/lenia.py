"""Lenia model."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from cax.core.ca import CA
from cax.types import State
from cax.utils.render import clip_and_uint8
from flax import nnx

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
		rule_params: RuleParams,
		state_size: int,
		state_scale: float = 1,
		kernel_fn: Callable = original_kernel_fn,
		growth_fn: Callable = growth_exponential,
	):
		"""Initialize Lenia."""
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
		update = LeniaUpdate(
			channel_size=channel_size,
			T=T,
			rule_params=rule_params,
			growth_fn=growth_fn,
		)

		super().__init__(perceive, update)

	@nnx.jit
	def update_rule_params(self, rule_params: RuleParams):
		"""Update the rule parameters."""
		self.perceive.update_rule_params(rule_params)
		self.update.update_rule_params(rule_params)

	@nnx.jit
	def render(self, state: State) -> jax.Array:
		"""Render state.

		Args:
			state: An array of states.

		Returns:
			Rendered states.

		"""
		assert self.num_dims == 2, "Lenia only supports 2D visualization."

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
