"""Perception base module for Cellular Automata.

This module defines the `Perceive` class, which serves as a base class for all perception modules
in the CAX library. The `Perceive` class inherits from `nnx.Module` to utilize Flax's neural
network functionalities. It is designed to be subclassed by specific perception implementations
that process the current state of the environment or system and return a perception.

"""

from flax import nnx

from cax.types import Perception, State


class Perceive(nnx.Module):
	"""A base class for perception modules.

	This class is designed to be subclassed by specific perception implementations.
	It inherits from flax.nnx.Module to leverage Flax's neural network functionalities.

	"""

	def __call__(self, state: State) -> Perception:
		"""Process the current state and return a perception.

		This method should be implemented by subclasses to define specific perception logic.

		Args:
			state: State of the cellular automaton.

		Returns:
			The resulting perception based on the input state.

		Raises:
			NotImplementedError: This base method raises an error if not overridden by subclasses.

		"""
		raise NotImplementedError
