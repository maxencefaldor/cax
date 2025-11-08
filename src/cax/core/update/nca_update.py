"""Neural Cellular Automata update module."""

from collections.abc import Callable, Sequence
from functools import partial

from flax import nnx
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception

from .residual_update import ResidualUpdate


class NCAUpdate(ResidualUpdate):
	"""Neural Cellular Automata update class.

	Builds on the residual update and applies an alive mask so that only active cells update.
	"""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		*,
		activation_fn: Callable = nnx.relu,
		step_size: float = 1.0,
		cell_dropout_rate: float = 0.0,
		kernel_size: Sequence[int] = (3, 3),
		alive_threshold: float = 0.1,
		zeros_init: bool = False,
		rngs: nnx.Rngs,
	):
		"""Initialize NCA update.

		Args:
			channel_size: Number of input channels.
			perception_size: Size of the perception.
			hidden_layer_sizes: Sizes of hidden layers.
			activation_fn: Activation function to use.
			step_size: Step size for the update.
			cell_dropout_rate: Dropout rate for cells.
			kernel_size: Size of the convolutional kernel.
			alive_threshold: Threshold for determining if a cell is alive.
			zeros_init: Whether to use zeros initialization for the weights of the last layer.
			rngs: rng key.

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
			zeros_init=zeros_init,
		)
		self.pool = partial(nnx.max_pool, window_shape=kernel_size, padding="SAME")
		self.alive_threshold = alive_threshold

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		Args:
			state: Current state.
			perception: Current perception.
			input: Optional input.

		Returns:
			Next state.

		"""
		alive_mask = self.get_alive_mask(state)
		state = super().__call__(state, perception, input)
		alive_mask &= self.get_alive_mask(state)
		return alive_mask * state

	def get_alive_mask(self, state: State) -> Array:
		"""Generate a mask of alive cells based on the current state.

		Args:
			state: Current state.

		Returns:
			A boolean mask indicating which cells are alive.

		"""
		alive = state_to_alive(state)
		alive_mask: Array = self.pool(alive) > self.alive_threshold
		return alive_mask


def state_to_alive(state: State) -> State:
	"""Extract the 'alive' component from the state.

	Args:
		state: Input state.

	Returns:
		The 'alive' component of the state.

	"""
	return state[..., -1:]
