"""Elementary Cellular Automata module.

This module implements Elementary Cellular Automata, one-dimensional cellular automata where each
cell's next state depends on its current state and that of its two immediate neighbors. Elementary
Cellular Automata are classified by Wolfram rule numbers (0-255), which define the transition
function for all possible three-cell neighborhoods.
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from .perceive import ElementaryPerceive
from .update import ElementaryUpdate


class Elementary(ComplexSystem):
	"""Elementary Cellular Automata class.

	A one-dimensional cellular automaton where each cell evolves based on its current state and the
	states of its two immediate neighbors according to a Wolfram rule. The system supports all 256
	possible rules and can simulate classic patterns such as Rule 30, Rule 110, and Rule 184.
	"""

	def __init__(
		self,
		*,
		wolfram_code: Array,
		rngs: nnx.Rngs,
	):
		"""Initialize Elementary Cellular Automaton.

		Args:
			wolfram_code: Array of 8 binary values defining the Wolfram rule. Each element
				corresponds to the output for one of the 8 possible three-cell neighborhood
				configurations (111, 110, 101, 100, 011, 010, 001, 000).
			rngs: rng key.

		"""
		self.perceive = ElementaryPerceive(rngs=rngs)
		self.update = ElementaryUpdate(wolfram_code=wolfram_code)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@classmethod
	def wolfram_code_from_rule_number(cls, rule_number: int) -> Array:
		"""Create Wolfram code array from a rule number.

		Converts a Wolfram rule number (0-255) to its binary representation as an array
		of 8 floats. For example, rule 30 becomes [0, 0, 0, 1, 1, 1, 1, 0].

		Args:
			rule_number: Integer between 0 and 255 representing the Wolfram rule.

		Returns:
			Array of shape (8,) containing binary values (0.0 or 1.0) representing
				the rule's lookup table.

		"""
		assert 0 <= rule_number < 256, "Wolfram code must be between 0 and 255."
		return ((rule_number >> 7 - jnp.arange(8)) & 1).astype(jnp.float32)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB image.

		Converts the one-dimensional cellular automaton state to an RGB visualization
		by replicating the single-channel state values across all three color channels,
		resulting in a grayscale image.

		Args:
			state: Array with shape (num_steps, width, 1) representing the
				cellular automaton state, where each cell contains a value in [0, 1].

		Returns:
			RGB image with dtype uint8 and shape (num_steps, width, 3), where cell values are mapped
				to grayscale colors in the range [0, 255].

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		return clip_and_uint8(rgb)
