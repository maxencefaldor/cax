"""Lenia update module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, State

from .growth import exponential_growth_fn
from .rule import LeniaRuleParams


class LeniaUpdate(Update):
	"""Lenia update class."""

	def __init__(
		self,
		channel_size: int,
		*,
		T: int,
		growth_fn: Callable = exponential_growth_fn,
		rule_params: LeniaRuleParams,
	):
		"""Initialize Lenia update.

		Args:
			channel_size: Number of channels.
			T: Time resolution.
			growth_fn: Growth mapping function.
			rule_params: Parameters for the rules.

		"""
		self.channel_size = channel_size
		self.T = T

		self.weight = rule_params.weight
		self.reshape_kernel_to_channel = self._reshape_kernel_to_channel(rule_params)

		self.growth_fn = growth_fn
		self.growth_params = nnx.data(rule_params.growth_params)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Lenia update.

		Args:
			state: Current state.
			perception: Perceived state.
			input: Input (unused in this implementation).

		Returns:
			Next state.

		"""
		# Compute growth
		G_k = self.weight * self.growth_fn(perception, self.growth_params)

		# Aggregate growth to channels
		G = jnp.dot(G_k, self.reshape_kernel_to_channel)

		# Update state and clip
		state = jnp.clip(state + G / self.T, 0.0, 1.0)

		return state

	def _reshape_kernel_to_channel(self, rule_params: LeniaRuleParams) -> Array:
		"""Compute array to reshape from kernel to channel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size))(
			rule_params.channel_target
		)
