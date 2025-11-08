"""Complex system module.

This module defines the abstract interface for complex systems simulated in CAX. A complex
system encapsulates state transition dynamics over discrete time steps and a rendering routine
to visualize states.

Subclasses must implement `_step` for a single-step transition and `render` for converting
a state to an RGB image representation. The public `__call__` method handles multi-step
evolution with JAX/Flax scanning utilities.

"""

from functools import partial
from typing import Any

from flax import nnx
from jax import Array

from cax.types import Input, State


class ComplexSystem(nnx.Module):
	"""Base class for complex systems.

	This class specifies the minimal interface for systems that evolve a `State` over time.
	It provides a JIT-compiled multi-step driver via `__call__` that wraps the subclass-defined
	single-step transition `_step`.

	Subclasses typically compose perception and update modules and may store hyperparameters
	and learned parameters within the Flax `nnx.Module` state.

	"""

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Step the system by a single time step.

		Implementations should be side-effect free with respect to the provided `state` argument
		(unless leveraging Flax `sow`/`nnx.Intermediate` mechanics) and return the next state.
		Shapes and dtypes of `state` are system-specific but should be stable across steps.

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
		"""Step the system for multiple time steps.

		This method wraps `_step` inside a JAX scan for efficiency and JIT-compiles the loop.
		If `input` is time-varying, set `input_in_axis` to the axis containing the time
		dimension so that each step receives the corresponding slice of input.

		Args:
			state: Current state.
			input: Optional input.
			num_steps: Number of steps.
			input_in_axis: Axis for input if provided for each step.
			sow: Whether to sow intermediate values.

		Returns:
			Final state after `num_steps` applications of `_step`.

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
	def render(self, state: State, **kwargs: Any) -> Array:
		"""Render state to RGB image.

		Implementations should return values in the range `[0, 255]` with dtype `uint8` and
		shape `(..., 3)` for RGB. For systems that naturally produce RGBA, either drop the alpha
		channel or composite it over a background in this method.

		Args:
			state: A state.
			**kwargs: Additional rendering-specific keyword arguments.

		Returns:
			An RGB image with dtype `uint8` and shape `(..., 3)`.

		"""
		raise NotImplementedError
