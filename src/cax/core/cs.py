"""Complex system module."""

from functools import partial

from flax import nnx
from jax import Array

from cax.types import Input, State


class ComplexSystem(nnx.Module):
	"""Complex system class."""

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Step the complex system for a single time step.

		Args:
			state: Current state.
			input: Optional input.
			sow: Whether to sow intermediate values.

		Returns:
			Next state.

		"""
		raise NotImplementedError

	@partial(nnx.jit, static_argnames=("num_steps", "input_in_axis", "sow"))
	def __call__(
		self,
		state: State,
		input: Input | None = None,
		*,
		num_steps: int = 1,
		input_in_axis: int | None = None,
		sow: bool = False,
	) -> State:
		"""Step the complex system for multiple time steps.

		Args:
			state: Current state.
			input: Optional input.
			num_steps: Number of steps.
			input_in_axis: Axis for input if provided for each step.
			sow: Whether to sow intermediate values.

		Returns:
			Final state.

		"""
		state_axes = nnx.StateAxes({nnx.Intermediate: 0, ...: nnx.Carry})
		state = nnx.scan(
			lambda cs, state, input: cs._step(state, input, sow=sow),
			in_axes=(state_axes, nnx.Carry, input_in_axis),
			out_axes=nnx.Carry,
			length=num_steps,
		)(self, state, input)

		return state

	@nnx.jit
	def render(self, state: State, **kwargs) -> Array:
		"""Render state to RGB.

		Args:
			state: A state.
			**kwargs: Additional rendering-specific keyword arguments.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		raise NotImplementedError
