"""Life update module."""

import jax.numpy as jnp
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update


class LifeUpdate(Update):
	"""Life update class."""

	def __init__(self, birth: Array, survival: Array):
		"""Initialize Life update.

		Args:
			birth: Birth rule.
			survival: Survival rule.

		"""
		self.birth = birth
		self.survival = survival

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Life rules based on birth/survival.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		self_alive = perception[..., 0:1]
		num_alive_neighbors = perception[..., 1:2].astype(jnp.int32)

		# Birth
		birth = jnp.logical_and(1.0 - self_alive, self.birth[num_alive_neighbors])

		# Survival
		survival = jnp.logical_and(self_alive, self.survival[num_alive_neighbors])

		# Combine the conditions for the next state
		state = jnp.where(birth | survival, 1.0, 0.0)
		return state
