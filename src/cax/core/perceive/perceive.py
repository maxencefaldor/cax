"""Perceive base module."""

from flax import nnx


class Perceive[State, Perception](nnx.Module):
	"""Base class for perception modules.

	Subclasses implement neighborhood gathering or convolutional transforms that map a state
	to a perception. Perceptions are PyTrees — commonly arrays shaped
	`(..., *spatial_dims, perception_size)`, or a dataclass of arrays — and the type
	parameter names which one a subclass produces, so the perceive-update handoff is
	checked instead of erased.
	"""

	def __call__(self, state: State) -> Perception:
		"""Process the current state to produce a perception.

		Args:
			state: Current state.

		Returns:
			Perception derived from `state`.

		"""
		raise NotImplementedError
