"""Life update module."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, Rule, State


class LifeUpdate(Update):
	"""Life update class."""

	def __init__(self, rngs: nnx.Rngs):
		"""Initialize LifeUpdate."""
		self.birth = Rule(jax.random.bernoulli(rngs(), shape=(9,)).astype(jnp.float32))
		self.survival = Rule(jax.random.bernoulli(rngs(), shape=(9,)).astype(jnp.float32))

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Life rules based on birth/survival.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

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

	@nnx.jit
	def update_birth_survival(self, birth: Array, survival: Array) -> None:
		"""Update the birth/survival."""
		self.birth.value = birth
		self.survival.value = survival

	def update_birth_survival_from_string(self, rule_string: str) -> None:
		"""Update the birth/survival rules from a rule string.

		Args:
			rule_string: A string in the format "B{birth_numbers}/S{survival_numbers}",
				where birth_numbers and survival_numbers are lists of digits.
				For example, "B3/S23" for Conway"s Game of Life.

		"""
		assert "/" in rule_string, (
			f"Invalid rule string format: {rule_string}. Expected format: B{{digits}}/S{{digits}}"
		)

		# Split the rule string into birth and survival parts
		birth_string, survival_string = rule_string.split("/")

		assert birth_string.startswith("B"), (
			f"Invalid rule string format: {rule_string}. Expected format: B{{digits}}/S{{digits}}"
		)
		assert survival_string.startswith("S"), (
			f"Invalid rule string format: {rule_string}. Expected format: B{{digits}}/S{{digits}}"
		)

		# Extract the birth and survival numbers
		birth_numbers = [int(digit) for digit in birth_string[1:]]
		survival_numbers = [int(digit) for digit in survival_string[1:]]

		assert all(0 <= num <= 8 for num in birth_numbers + survival_numbers), (
			"Numbers in rule string must be between 0 and 8."
		)

		# Update the birth and survival rules
		birth = jnp.array(
			[1.0 if num_neighbors in birth_numbers else 0.0 for num_neighbors in range(9)]
		)
		survival = jnp.array(
			[1.0 if num_neighbors in survival_numbers else 0.0 for num_neighbors in range(9)]
		)
		self.update_birth_survival(birth, survival)
