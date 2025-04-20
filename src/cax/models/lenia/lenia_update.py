"""Lenia update module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, State

from .growth import exponential_growth_fn
from .rule import RuleParams


class LeniaUpdate(Update):
	"""Lenia update class."""

	def __init__(
		self,
		channel_size: int,
		T: int,
		rule_params: RuleParams,
		*,
		growth_fn: Callable = exponential_growth_fn,
	):
		"""Initialize the LeniaUpdate.

		Args:
			channel_size: Number of channels.
			T: Time resolution.
			rule_params: Parameters for the rules.
			growth_fn: Growth function.

		"""
		super().__init__()
		self.channel_size = channel_size
		self.T = T
		self.growth_fn = growth_fn

		# Set rule parameters
		self.rule_params = jax.tree.map(nnx.Param, rule_params)

		# Reshape kernel to channel
		self.reshape_kernel_to_channel = nnx.Param(
			self.compute_reshape_kernel_to_channel(rule_params)
		)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Lenia update rule.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state.
			input: External input (unused in this implementation).

		Returns:
			Updated state after applying the Lenia.

		"""
		# Compute growth
		G_k = self.rule_params.weight * self.growth_fn(perception, self.rule_params.growth_params)

		# Aggregate growth to channels
		G = jnp.dot(G_k, self.reshape_kernel_to_channel.value)

		# Update state and clip
		state = jnp.clip(state + G / self.T, 0.0, 1.0)

		return state

	@nnx.jit
	def update_rule_params(self, rule_params: RuleParams):
		"""Update the rule parameters."""
		# Update rule parameters
		self.rule_params = jax.tree.map(nnx.Param, rule_params)

		# Compute reshape channel to kernel
		self.reshape_kernel_to_channel.value = self.compute_reshape_kernel_to_channel(rule_params)

	@nnx.jit
	def compute_reshape_kernel_to_channel(self, rule_params: RuleParams) -> Array:
		"""Compute array to reshape from kernel to channel."""
		return nnx.vmap(lambda x: jax.nn.one_hot(x, num_classes=self.channel_size))(
			rule_params.channel_target
		)
