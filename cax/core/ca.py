"""Cellular Automata module."""

from functools import partial

from flax import nnx

from cax.core.perceive.perceive import Perceive
from cax.core.update.update import Update
from cax.types import Input, State


class CA(nnx.Module):
	"""Cellular Automata class."""

	perceive: Perceive
	update: Update

	def __init__(self, perceive: Perceive, update: Update):
		"""Initialize the CA.

		Args:
			perceive: Perception module.
			update: Update module.

		"""
		self.perceive = perceive
		self.update = update

	@nnx.jit
	def step(self, state: State, input: Input | None = None) -> State:
		"""Perform a single step of the CA.

		Args:
			state: Current state.
			input: Optional input.

		Returns:
			Updated state.

		"""
		perception = self.perceive(state)
		state = self.update(state, perception, input)
		return state

	@partial(nnx.jit, static_argnames=("num_steps", "all_steps", "input_in_axis"))
	def __call__(
		self,
		state: State,
		input: Input | None = None,
		*,
		num_steps: int = 1,
		all_steps: bool = False,
		input_in_axis: int | None = None,
	) -> State:
		"""Run the CA for multiple steps.

		Args:
			state: Initial state.
			input: Optional input.
			num_steps: Number of steps to run.
			all_steps: Whether to return all intermediate states.
			input_in_axis: Axis for input if provided for each step.

		Returns:
			Final state or all intermediate states if all_steps is True.

		"""

		def step(carry: tuple[CA, State], input: Input | None) -> tuple[tuple[CA, State], State]:
			ca, state = carry
			state = ca.step(state, input)
			return (ca, state), state if all_steps else None  # type: ignore

		(_, state), states = nnx.scan(
			step,
			in_axes=(nnx.Carry, input_in_axis),
			length=num_steps,
		)((self, state), input)

		return states if all_steps else state
