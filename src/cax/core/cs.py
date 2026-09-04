"""Complex system module.

This module defines the abstract interface for complex systems simulated in CAX. A
complex system encapsulates state transition dynamics over discrete time steps and a
rendering routine to visualize states.

Subclasses must implement `_step` for a single-step transition and `render` for
converting a state to an RGB image representation. The public `__call__` method handles
multi-step evolution with JAX/Flax scanning utilities and returns the final state; under
`return_states=True` it also returns the per-step states as the scan's stacked outputs.

"""

from typing import Any

from flax import nnx
from jax import Array


class ComplexSystem[State, Input](nnx.Module):
    """Base class for complex systems.

    This class specifies the minimal interface for systems that evolve a `State` over
    time. It provides a JIT-compiled multi-step driver via `__call__` that wraps the
    subclass-defined single-step transition `_step`.

    Subclasses typically compose perception and update modules and may store
    hyperparameters and learned parameters within the Flax `nnx.Module` state.

    Attributes:
        remat: If True, applies gradient checkpointing (rematerialization) to the scan
            body, trading compute for memory during backpropagation through long step
            sequences. Subclasses can set this as a class variable or instance
            attribute.

    """

    remat: bool = False

    def _step(self, state: State, input: Input | None = None) -> State:
        """Step the system by a single time step.

        Implementations should be pure with respect to the provided `state` argument and
        return the next state. Shapes and dtypes of `state` are system-specific but
        should be stable across steps.

        Args:
            state: Current state.
            input: Optional input.

        Returns:
            Next state.

        """
        raise NotImplementedError

    @nnx.jit(static_argnames=("num_steps", "input_in_axis", "return_states"))
    def __call__(
        self,
        state: State,
        input: Input | None = None,
        *,
        num_steps: int = 1,
        input_in_axis: int | None = None,
        return_states: bool = False,
    ) -> State | tuple[State, State]:
        """Step the system for multiple time steps.

        This method wraps `_step` inside a JAX scan for efficiency and JIT-compiles the
        loop. If `input` is time-varying, set `input_in_axis` to the axis containing the
        time dimension so that each step receives the corresponding slice of input.

        Under `return_states=True`, the per-step states are also returned as the scan's
        stacked outputs, mirroring the `(carry, ys)` convention of `jax.lax.scan`. The
        trajectory holds the state *after* each step, stacked along a new leading axis
        of size `num_steps` — its first element is the state after one step, its last
        equals the final state, and the initial state is not included.

        When `remat` is enabled, the scan body is wrapped with `nnx.remat` to reduce
        memory usage during backpropagation at the cost of recomputing intermediates.

        Note that `num_steps`, `input_in_axis`, and `return_states` are static: each
        distinct combination compiles once, so sweeps over horizons should batch their
        step counts.

        Args:
            state: Current state.
            input: Optional input.
            num_steps: Number of steps.
            input_in_axis: Axis for input if provided for each step.
            return_states: Whether to also return the stacked per-step states.

        Returns:
            Final state after `num_steps` applications of `_step`, or a
                `(final_state, states)` tuple under `return_states=True`, where `states`
                stacks the per-step states along a new leading axis of size `num_steps`.

        """

        def step_fn(
            cs: ComplexSystem[State, Input], state: State, input: Input | None
        ) -> tuple[State, State | None]:
            next_state = cs._step(state, input)
            return next_state, (next_state if return_states else None)

        if self.remat:
            step_fn = nnx.remat(step_fn)

        state, states = nnx.scan(
            step_fn,
            in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, input_in_axis),
            out_axes=(nnx.Carry, 0),
            length=num_steps,
        )(self, state, input)

        return (state, states) if return_states else state

    @nnx.jit
    def render(self, state: State, **kwargs: Any) -> Array:
        """Render state to RGB image.

        Implementations should return values in the range `[0, 255]` with dtype `uint8`
        and shape `(..., 3)` for RGB. For systems that naturally produce RGBA, either
        drop the alpha channel or composite it over a background in this method.

        Args:
            state: A state.
            **kwargs: Additional rendering-specific keyword arguments.

        Returns:
            An RGB image with dtype `uint8` and shape `(..., 3)`.

        """
        raise NotImplementedError
