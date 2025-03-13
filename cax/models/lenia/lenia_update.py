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

	rules_params: RuleParams
	reshape_kernel_to_channel: nnx.Variable

	def __init__(
		self,
		T: int,
		state_scale: float,
		rules_params: RuleParams,
		growth_fn: Callable = growth_exponential,
		flow: bool = False,
	):
		"""Initialize the LeniaUpdate.

		Args:
			T: Time resolution.
			state_scale: Integer to scale the state.
			rules_params: Parameters for the rules.
			growth_fn: Growth function.
			flow: Whether to use flow or not.

		"""
		super().__init__()
		self.T = T
		self.state_scale = state_scale
		self.rules_params = jax.tree.map(nnx.Variable, rules_params)
		self.growth_fn = growth_fn
		self.flow = flow

		self.reshape_kernel_to_channel = nnx.Param(
			jax.vmap(lambda x: jax.nn.one_hot(x, num_classes=3))(rules_params.channel_target)
		)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Lenia update rule with flow.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state.
			input: External input (unused in this implementation).

		Returns:
			Updated state after applying the Lenia rule with flow.

		"""
		# Original Lenia update
		g_k = self.rules_params.weight[None, None, ...] * self.growth_fn(
			perception, self.rules_params.growth_params
		)  # (y, x, k,)
		g = jnp.dot(g_k, self.reshape_kernel_to_channel.value)  # (y, x, c,)
		state = jnp.clip(state + 1 / self.T * g, 0.0, 1.0)  # (y, x, c,)

		return state
