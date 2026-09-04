"""Update base module."""

from flax import nnx


class Update[State, Perception, Input](nnx.Module):
    """Base class for update modules.

    Subclasses implement transforms mapping a state and a perception (and optional
    input) to the next state.
    """

    def __call__(
        self, state: State, perception: Perception, input: Input | None = None
    ) -> State:
        """Process the current state, perception, and input to produce a new state.

        This method should be implemented by subclasses to define specific update logic.

        Args:
            state: Current state.
            perception: Current perception.
            input: Optional input.

        Returns:
            Next state.

        """
        raise NotImplementedError
