"""Langton's Ant module.

This module implements Langton's Ant and its multi-color generalizations. Langton's Ant is a
two-dimensional cellular automaton with a mobile agent (the "ant") that moves on a grid, turning
and flipping cell colors according to a rule string. Despite its simple definition, the system
exhibits complex emergent behavior — notably the formation of a "highway" after approximately
10,000 steps in the classic RL variant.
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input
from cax.utils import clip_and_uint8, hsv_to_rgb

from .perceive import LangtonAntPerceive
from .state import LangtonAntState
from .update import LangtonAntUpdate

TURN_CHAR_TO_INT = {"N": 0, "R": 1, "U": 2, "L": 3}


class LangtonAnt(ComplexSystem):
	"""Langton's Ant and multi-color generalizations.

	A two-dimensional cellular automaton with a mobile agent that traverses a grid of colored
	cells. At each step the ant turns according to the color of its current cell, advances the
	cell to the next color in a cyclic sequence, and moves forward one step. The rule is
	specified as a sequence of turn directions — one per color — encoded as a string over the
	alphabet {R, L, N, U}.

	Classic examples include "RL" (Langton's original ant, produces a highway), "LLRR"
	(symmetric growth filling a square), and "LRRRRRLLR" (triangle-building ant).
	"""

	def __init__(self, *, turns: Array, rngs: nnx.Rngs):
		"""Initialize Langton's Ant.

		Args:
			turns: Array of shape (num_colors,) with integer values in {0, 1, 2, 3}
				encoding turn directions for each cell color. Values correspond to
				clockwise rotation in multiples of 90 degrees: 0=no turn, 1=right,
				2=u-turn, 3=left.
			rngs: rng key.

		"""
		self.perceive = LangtonAntPerceive()
		self.update = LangtonAntUpdate(turns=turns)

	def _step(
		self, state: LangtonAntState, input: Input | None = None, *, sow: bool = False
	) -> LangtonAntState:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@classmethod
	def turns_from_rule_string(cls, rule_string: str) -> Array:
		"""Create turns array from a rule string.

		Parses a rule string in the standard Langton's Ant notation where each character
		specifies the turn direction for the corresponding cell color.

		Args:
			rule_string: String of characters from {R, L, N, U}. For example, "RL" for
				classic Langton's Ant. The length of the string determines the number of
				cell colors.

		Returns:
			Array of shape (num_colors,) with integer turn values where 0=no turn,
				1=right (90 degrees clockwise), 2=u-turn (180 degrees), 3=left (90 degrees
				counter-clockwise).

		"""
		assert len(rule_string) >= 2, (
			f"Rule string must have at least 2 characters, got: '{rule_string}'"
		)

		valid_chars = set(TURN_CHAR_TO_INT.keys())
		for char in rule_string:
			assert char in valid_chars, (
				f"Invalid character '{char}' in rule string '{rule_string}'. "
				f"Valid characters are: {sorted(valid_chars)}"
			)

		turns = jnp.array([TURN_CHAR_TO_INT[char] for char in rule_string], dtype=jnp.int32)
		return turns

	@nnx.jit
	def render(self, state: LangtonAntState) -> Array:
		"""Render state to RGB image.

		Visualizes the grid by mapping each cell color to a distinct hue. For 2-color rules
		the rendering is grayscale (black for color 0, white for color 1). For multi-color
		rules, color 0 (unvisited) is black and colors 1 through num_colors-1 are evenly
		spaced around the HSV hue wheel at full saturation and value.

		Args:
			state: Langton's Ant state with grid of shape (height, width, 1).

		Returns:
			RGB image with dtype uint8 and shape (height, width, 3).

		"""
		num_colors = self.update.turns.shape[0]

		if num_colors == 2:
			rgb = jnp.repeat(state.grid, 3, axis=-1)
		else:
			visited = state.grid[..., 0:1] > 0
			hue = jnp.where(visited, (state.grid[..., 0:1] - 1) / (num_colors - 1), 0.0)
			saturation = jnp.where(visited, 1.0, 0.0)
			value = jnp.where(visited, 1.0, 0.0)
			hsv = jnp.concatenate([hue, saturation, value], axis=-1)
			rgb = hsv_to_rgb(hsv)

		return clip_and_uint8(rgb)
