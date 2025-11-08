"""Perceive base module."""

from flax import nnx

from cax.core import State
from cax.types import PyTree

Perception = PyTree


class Perceive(nnx.Module):
	"""Base class for perception modules.

	Subclasses implement neighborhood gathering or convolutional transforms that map a state
	to a perception. Perceptions are PyTrees; commonly arrays shaped
	`(..., *spatial_dims, perception_size)`.
	"""

	def __call__(self, state: State) -> Perception:
		"""Process the current state to produce a perception.

		Args:
			state: Current state.

		Returns:
			Perception derived from `state`.

		"""
		raise NotImplementedError
