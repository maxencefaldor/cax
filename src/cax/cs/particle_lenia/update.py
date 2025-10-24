"""Particle Lenia update module."""

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ParticleLeniaUpdate(Update):
	"""Particle Lenia update class."""

	def __init__(
		self,
		*,
		T: int,
	):
		"""Initialize Particle Lenia update.

		Args:
			T: Time resolution.

		"""
		self.T = T

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply Particle Lenia update.

		Args:
			state: Current state.
			perception: Perceived state.
			input: Input (unused in this implementation).

		Returns:
			Next state.

		"""
		state = state + perception / self.T
		return state
