"""Neural Cellular Automata update module."""

from collections.abc import Callable, Sequence
from functools import partial

from flax import nnx
from jax import Array

from cax.core.state import state_to_alive
from cax.core.update.residual_update import ResidualUpdate
from cax.types import Input, Perception, State


class NCAUpdate(ResidualUpdate):
	"""Neural Cellular Automata Update class."""

	alive_threshold: float

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		rngs: nnx.Rngs,
		*,
		activation_fn: Callable = nnx.relu,
		step_size: float = 1.0,
		cell_dropout_rate: float = 0.0,
		kernel_size: Sequence[int] = (3, 3),
		alive_threshold: float = 0.1,
	):
		"""Initialize the NCAUpdate layer.

		Args:
			channel_size: Number of input channels.
			perception_size: Size of the perception.
			hidden_layer_sizes: Sizes of hidden layers.
			rngs: Random number generators.
			activation_fn: Activation function to use.
			step_size: Step size for the update.
			cell_dropout_rate: Dropout rate for cells.
			kernel_size: Size of the convolutional kernel.
			alive_threshold: Threshold for determining if a cell is alive.

		"""
		super().__init__(
			num_spatial_dims=len(kernel_size),
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			rngs=rngs,
			activation_fn=activation_fn,
			step_size=step_size,
			cell_dropout_rate=cell_dropout_rate,
		)
		self.pool = partial(nnx.max_pool, window_shape=kernel_size, padding="SAME")
		self.alive_threshold = alive_threshold

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the NCA update to the input state.

		Args:
			state: Current state of the cellular automata.
			perception: Perceived state.
			input: External input to the system.

		Returns:
			Updated state after applying the NCA rules.

		"""
		alive_mask = self.get_alive_mask(state)
		state = super().__call__(state, perception, input)
		alive_mask &= self.get_alive_mask(state)
		return alive_mask * state

	def get_alive_mask(self, state: State) -> Array:
		"""Generate a mask of alive cells based on the current state.

		Args:
			state: Current state of the cellular automata.

		Returns:
			A boolean mask indicating which cells are alive.

		"""
		alive = state_to_alive(state)
		alive_mask: Array = self.pool(alive) > self.alive_threshold
		return alive_mask
