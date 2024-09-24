"""Residual update module with cell dropout."""

from collections.abc import Callable

from flax import nnx

from cax.core.update.mlp_update import MLPUpdate
from cax.types import Input, Perception, State


class ResidualUpdate(MLPUpdate):
	"""Residual update module with cell dropout."""

	step_size: float
	cell_dropout_rate: float

	def __init__(
		self,
		num_spatial_dims: int,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		rngs: nnx.Rngs,
		*,
		activation_fn: Callable = nnx.relu,
		step_size: float = 1.0,
		cell_dropout_rate: float = 0.0,
	):
		"""Initialize the ResidualUpdate module.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			channel_size: Number of channels in the state.
			perception_size: Size of the perception input.
			hidden_layer_sizes: Sizes of hidden layers in the MLP.
			rngs: Random number generators.
			activation_fn: Activation function to use.
			step_size: Step size for the residual update.
			cell_dropout_rate: Dropout rate for cell updates.

		"""
		super().__init__(
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			rngs=rngs,
			activation_fn=activation_fn,
			num_spatial_dims=num_spatial_dims,
		)
		self.dropout = nnx.Dropout(rate=cell_dropout_rate, broadcast_dims=(-1,), rngs=rngs)
		self.step_size = step_size

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the residual update to the state.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived information from the environment.
			input: External input to the system.

		Returns:
			Updated state after applying the residual update.

		"""
		update = super().__call__(state, perception, input)
		update = self.dropout(update)
		state += self.step_size * update
		return state
