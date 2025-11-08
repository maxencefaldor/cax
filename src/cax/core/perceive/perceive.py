"""Perceive base module.

This module defines the `Perceive` class, which serves as a base class for all perception modules
in the CAX library. The `Perceive` class inherits from `nnx.Module` to utilize Flax's neural
network functionalities. It is designed to be subclassed by specific perception implementations
that process the current state of the environment or system and return a perception.
"""

from flax import nnx

from cax.core import State
from cax.types import PyTree

Perception = PyTree


class Perceive(nnx.Module):
	"""A base class for perception modules.

	This class is designed to be subclassed by specific perception implementations.
	It inherits from flax.nnx.Module to leverage Flax's neural network functionalities.

	"""

	def __call__(self, state: State) -> Perception:
		"""Process the current state to produce a perception.

		This method should be implemented by subclasses to define specific perception logic.

		Args:
			state: Current state.

		Returns:
			Perception based on the current state.

		"""
		raise NotImplementedError
