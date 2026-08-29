"""Elementary Cellular Automata update module.

This module implements the update rule for Elementary Cellular Automata based on Wolfram codes.
Each cell's next state is determined by converting its three-cell neighborhood to a 3-bit
index and looking up the result in the Wolfram code table.

"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.update import Update


class ElementaryUpdate(Update[Array, Array, Array]):
	"""Elementary Cellular Automata update rule.

	Applies the Wolfram rule by converting each cell's three-cell neighborhood to a
	binary index and looking up the corresponding output in the rule table.

	"""

	def __init__(self, *, wolfram_code: Array):
		"""Initialize Elementary update.

		Args:
			wolfram_code: Array of 8 binary values defining the Wolfram rule. Each element
				corresponds to the output for one of the 8 possible three-cell neighborhood
				configurations (111, 110, 101, 100, 011, 010, 001, 000).

		"""
		self.lut = wolfram_code[::-1]

	@override
	def __call__(self, state: Array, perception: Array, input: Array | None = None) -> Array:
		"""Process the current state, perception, and input to produce a new state.

		Converts each cell's three-cell neighborhood to a 3-bit index (left*4 + self*2 + right)
		and looks up the corresponding output in the Wolfram rule table.

		Args:
			state: Current state (unused, next state computed solely from perception).
			perception: Array with shape (..., width, 3) containing the three-cell neighborhood
				in [self, right, left] order.
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (..., width, 1) containing the updated cell values.

		"""
		index = (perception[..., 2:3] * 4 + perception[..., 0:1] * 2 + perception[..., 1:2]).astype(
			jnp.int32
		)
		return self.lut[index[..., 0]][..., None]
