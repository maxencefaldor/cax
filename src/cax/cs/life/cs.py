"""Life module.

This module implements Conway's Game of Life and other Life-like cellular automata
that follow birth/survival rules. These are two-dimensional cellular automata where
each cell's next state depends on its current state and the number of alive neighbors
in its Moore neighborhood (8 surrounding cells).
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from .perceive import LifePerceive
from .update import LifeUpdate


class Life(ComplexSystem):
	"""Conway's Game of Life and Life-like cellular automata.

	A two-dimensional cellular automaton where each cell evolves based on its current
	state (alive or dead) and the number of alive neighbors in its Moore neighborhood.
	The system is defined by birth and survival rules that determine when cells become
	alive or remain alive. Classic examples include Conway's Game of Life (B3/S23),
	HighLife (B36/S23), and Day & Night (B3678/S34678).
	"""

	def __init__(
		self,
		*,
		birth: Array,
		survival: Array,
		rngs: nnx.Rngs,
	):
		"""Initialize Life.

		Args:
			birth: Array of shape (9,) defining birth conditions. Element i is 1.0 if a dead
				cell with i alive neighbors should become alive, 0.0 otherwise.
			survival: Array of shape (9,) defining survival conditions. Element i is 1.0 if a
				live cell with i alive neighbors should stay alive, 0.0 otherwise.
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
		"""Create birth and survival arrays from a rule string in Golly format.

		Parses a rule string in the standard B/S notation used by Golly and other
		Life simulators. For example, "B3/S23" represents Conway's Game of Life,
		where dead cells with exactly 3 neighbors become alive (Birth), and live
		cells with 2 or 3 neighbors survive (Survival).

		Args:
			rule_golly: Rule string in format "B{birth_numbers}/S{survival_numbers}",
				where birth_numbers and survival_numbers are digits from 0 to 8.
				For example, "B3/S23" for Conway's Game of Life.

		Returns:
			Tuple of (birth, survival) arrays, each of shape (9,) containing binary
				values (0.0 or 1.0) indicating which neighbor counts activate the rule.

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
		"""Render state to RGB image.

		Converts the Life state to an RGB visualization by replicating the single-channel
		state values across all three color channels, resulting in a grayscale image where
		alive cells appear white and dead cells appear black.

		Args:
			state: Array with shape (..., height, width, 1) representing the Life state,
				where each cell is 0.0 (dead) or 1.0 (alive).

		Returns:
			RGB image with dtype uint8 and shape (..., height, width, 3), where cell
				values are mapped to grayscale colors in the range [0, 255].

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		return clip_and_uint8(rgb)
