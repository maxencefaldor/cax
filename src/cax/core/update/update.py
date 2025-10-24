"""Update base module.

This module defines the `Update` class, which serves as a base class for all update modules
in the CAX library. The `Update` class inherits from `nnx.Module` to utilize Flax's neural
network functionalities.
"""

from flax import nnx

from cax.types import Input, Perception, State


class Update(nnx.Module):
	"""A base class for update modules.

	This class is designed to be subclassed by specific update implementations.
	It inherits from flax.nnx.Module to leverage Flax's neural network functionalities.
	"""

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		This method should be implemented by subclasses to define specific update logic.

		Args:
			state: Current state.
			perception: Current perception.
			input: Optional input.

		Returns:
			Next state.

		"""
		raise NotImplementedError
