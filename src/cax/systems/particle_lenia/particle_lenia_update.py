"""Particle Lenia update module."""

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class ParticleLeniaUpdate(Update):
	"""Particle Lenia update class."""

	def __init__(
		self,
		T: int,
	):
		"""Initialize the LeniaUpdate.

		Args:
			T: Time resolution.

		"""
		super().__init__()
		self.T = T

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the Lenia update rule.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state.
			input: External input (unused in this implementation).

		Returns:
			Updated state after applying the Lenia.

		"""
		state = state + 1 / self.T * perception
		return state
