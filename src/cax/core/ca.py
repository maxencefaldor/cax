"""Cellular Automata module."""

from collections.abc import Callable
from functools import partial

from flax import nnx
from jax import Array

from cax.core.perceive import Perceive
from cax.core.update import Update
from cax.types import Input, Metrics, Perception, State


def metrics_fn(next_state: State, state: State, perception: Perception, input: Input) -> Metrics:
	"""Metrics function returning the state.

	Args:
		next_state: Next state.
		state: Current state.
		perception: Perception.
		input: Input.

	Returns:
		A PyTree of metrics.

	"""
	return next_state


class CA(nnx.Module):
	"""Cellular Automata class."""

	def __init__(self, perceive: Perceive, update: Update, *, metrics_fn: Callable = metrics_fn):
		"""Initialize the CA.

		Args:
			perceive: Perception module.
			update: Update module.
			metrics_fn: Metrics function.

		"""
		self.perceive = perceive
		self.update = update
		self.metrics_fn = metrics_fn

	@nnx.jit
	def step(self, state: State, input: Input | None = None) -> tuple[State, Metrics]:
		"""Perform a single step.

		Args:
			state: Current state.
			input: Optional input.

		Returns:
			Updated state.

		"""
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)
		return next_state, self.metrics_fn(next_state, state, perception, input)

	@partial(nnx.jit, static_argnames=("num_steps", "input_in_axis"))
	def __call__(
		self,
		state: State,
		input: Input | None = None,
		*,
		num_steps: int = 1,
		input_in_axis: int | None = None,
	) -> tuple[State, Metrics]:
		"""Run the CA for multiple steps.

		Args:
			state: Initial state.
			input: Optional input.
			num_steps: Number of steps to run.
			input_in_axis: Axis for input if provided for each step.

		Returns:
			Final state and all intermediate metrics.

		"""

		def step(carry: tuple[CA, State], input: Input | None) -> tuple[tuple[CA, State], State]:
			ca, state = carry
			state, metrics = ca.step(state, input)
			return (ca, state), metrics

		(_, state), metrics = nnx.scan(
			step,
			in_axes=(nnx.Carry, input_in_axis),
			length=num_steps,
		)((self, state), input)

		return state, metrics

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		raise NotImplementedError
