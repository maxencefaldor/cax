"""Life module."""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from .perceive import LifePerceive
from .update import LifeUpdate


class Life(ComplexSystem):
	"""Life class."""

	def __init__(
		self,
		*,
		birth: Array,
		survival: Array,
		rngs: nnx.Rngs,
	):
		"""Initialize Life.

		Args:
			birth: Birth rule.
			survival: Survival rule.
			rngs: rng key.

		"""
		self.perceive = LifePerceive(rngs=rngs)
		self.update = LifeUpdate(birth=birth, survival=survival)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@classmethod
	def birth_survival_from_string(cls, rule_golly: str) -> tuple[Array, Array]:
		"""Create birth and survival arrays from a rule in Golly format.

		Args:
			rule_golly: A string in the format "B{birth_numbers}/S{survival_numbers}",
				where birth_numbers and survival_numbers are lists of digits.
				For example, "B3/S23" for Conway's Game of Life.

		Returns:
			Birth and survival arrays.

		"""
		assert "/" in rule_golly, (
			f"Invalid rule string format: {rule_golly}. Expected format: B{{digits}}/S{{digits}}"
		)

		# Split the rule string into birth and survival parts
		birth_string, survival_string = rule_golly.split("/")

		assert birth_string.startswith("B"), (
			f"Invalid rule string format: {rule_golly}. Expected format: B{{digits}}/S{{digits}}"
		)
		assert survival_string.startswith("S"), (
			f"Invalid rule string format: {rule_golly}. Expected format: B{{digits}}/S{{digits}}"
		)

		# Extract the birth and survival numbers
		birth_numbers = [int(digit) for digit in birth_string[1:]]
		survival_numbers = [int(digit) for digit in survival_string[1:]]

		assert all(0 <= num <= 8 for num in birth_numbers + survival_numbers), (
			"Numbers in rule string must be between 0 and 8."
		)

		# Create birth and survival rules
		birth = jnp.array(
			[1.0 if num_neighbors in birth_numbers else 0.0 for num_neighbors in range(9)]
		)
		survival = jnp.array(
			[1.0 if num_neighbors in survival_numbers else 0.0 for num_neighbors in range(9)]
		)
		return birth, survival

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		return clip_and_uint8(rgb)
