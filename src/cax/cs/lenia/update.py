"""Lenia update module.

This module implements the update rule for Lenia, which applies a growth mapping to the
potential fields and updates the state. Growth values are computed per-kernel using a
parameterized growth function, then aggregated back to channels.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update

from .growth import exponential_growth_fn
from .rule import LeniaRuleParams


class LeniaUpdate(Update):
	"""Lenia update rule.

	Applies the growth mapping to potential fields to determine how much each cell should
	grow or decay. Growth is computed per-kernel using parameterized growth functions,
	weighted, and aggregated to target channels. The state is updated in discrete time
	steps with temporal resolution T.
	"""

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
			T: Time resolution controlling the temporal discretization. Higher values
				produce smoother temporal dynamics with smaller update steps.
			growth_fn: Callable that maps neighborhood potential to growth values. Defines
				how cells respond to their local environment.
			rule_params: Instance of LeniaRuleParams containing kernel and growth parameters
				for each channel.

		"""
		self.channel_size = channel_size
		self.T = T

		self.weight = rule_params.weight
		self.reshape_kernel_to_channel = self._reshape_kernel_to_channel(rule_params)

		self.growth_fn = growth_fn
		self.growth_params = nnx.data(rule_params.growth_params)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		Computes growth values from potential fields using the growth function, aggregates
		them to target channels, and updates the state with temporal resolution T. The
		updated state is clipped to [0, 1].

		Args:
			state: Array with shape (*spatial_dims, channel_size) representing the current state.
			perception: Array with shape (*spatial_dims, num_kernels) containing potential fields
				from the perception step.
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (*spatial_dims, channel_size) after applying growth and
				clipping to [0, 1].

		"""
		# Compute growth
		G_k = self.weight * self.growth_fn(perception, self.growth_params)

		# Aggregate growth to channels
		G = jnp.dot(G_k, self.reshape_kernel_to_channel)

		# Update state and clip
		state = jnp.clip(state + G / self.T, 0.0, 1.0)

		return state

	def _reshape_kernel_to_channel(self, rule_params: LeniaRuleParams) -> Array:
		"""Compute array to reshape from kernel to channel.

		Returns a matrix `K -> C` that aggregates per-kernel growth into channel space
		using `rule_params.channel_target`.

		Args:
			rule_params: Rule parameters containing the `channel_target` mapping.

		Returns:
			Array with shape `(K, C)` suitable for `jnp.dot(G_k, reshape)`.

		"""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size))(
			rule_params.channel_target
		)
