"""Lenia update module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from cax.core.update.update import Update
from cax.models.lenia.lenia_perceive import bell
from cax.types import Input, Perception, State
from flax import nnx
from jax import Array

from .types import GrowthParams, RuleParams


def growth_exponential(u: Array, growth_params: GrowthParams) -> Array:
	"""Growth mapping function introduced in [1]."""
	return 2 * bell(u, growth_params.mean, growth_params.std) - 1


class LeniaUpdate(Update):
	"""Lenia update class."""

	rule_params: RuleParams
	reshape_kernel_to_channel: nnx.Variable

	def __init__(
		self,
		channel_size: int,
		T: int,
		rule_params: RuleParams,
		growth_fn: Callable = growth_exponential,
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
		G_k = self.rule_params.weight[None, None, ...] * self.growth_fn(
			perception, self.rule_params.growth_params
		)  # (y, x, k,)

		# Aggregate growth to channels
		G = jnp.dot(G_k, self.reshape_kernel_to_channel.value)  # (y, x, c,)

		# Update state and clip
		state = jnp.clip(state + 1 / self.T * G, 0.0, 1.0)  # (y, x, c,)

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
