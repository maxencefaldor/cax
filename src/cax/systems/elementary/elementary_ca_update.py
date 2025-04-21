"""Elementary Cellular Automata update module."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.update.update import Update
from cax.types import Input, Perception, Rule, State


class ElementaryCAUpdate(Update):
	"""Elementary Cellular Automata update class."""

	def __init__(self, rngs: nnx.Rngs):
		"""Initialize the ElementaryUpdate."""
		self.rngs = rngs
		self.configurations = nnx.Param(
			jnp.array(
				[
					[1.0, 1.0, 1.0],
					[1.0, 1.0, 0.0],
					[1.0, 0.0, 1.0],
					[1.0, 0.0, 0.0],
					[0.0, 1.0, 1.0],
					[0.0, 1.0, 0.0],
					[0.0, 0.0, 1.0],
					[0.0, 0.0, 0.0],
				]
			)
		)
		self.wolfram_code = Rule(self.sample_wolfram_code())

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Elementary Cellular Automata update rule.

		Args:
			state: The current state of the cellular automaton.
			perception: The perceived state of the neighborhood.
			input: Additional input to the update rule (not used in this implementation).

		Returns:
			The updated state of the cellular automaton.

		"""

		def update_pattern(pattern: Array, value: Array) -> Array:
			return jnp.where(jnp.all(perception == pattern, axis=-1, keepdims=True), value, 0.0)

		state = jnp.sum(
			jax.vmap(update_pattern)(self.configurations.value, self.wolfram_code.value), axis=0
		)
		return state

	@nnx.jit
	def sample_wolfram_code(self) -> Array:
		"""Sample random Wolfram code."""
		return jax.random.bernoulli(self.rngs(), shape=(8,)).astype(jnp.float32)

	@nnx.jit
	def update_wolfram_code(self, wolfram_code: Array) -> None:
		"""Update the Wolfram code."""
		self.wolfram_code.value = wolfram_code

	def update_wolfram_code_from_string(self, wolfram_code: str) -> None:
		"""Set the rule from a string of 8 bits."""
		assert len(wolfram_code) == 8, "Wolfram code must be 8 bits long."
		assert all(bit in "01" for bit in wolfram_code), "Wolfram code must contain only 0s and 1s."
		self.wolfram_code.value = jnp.array([int(bit) for bit in wolfram_code], dtype=jnp.float32)
