"""Residual update module."""

from collections.abc import Callable

from flax import nnx

from cax.core import Input, State
from cax.core.perceive import Perception

from .mlp_update import MLPUpdate


class ResidualUpdate(MLPUpdate):
	"""Residual update class.

	Extends the MLP update with a residual connection and cell dropout applied to the update.
	"""

	def __init__(
		self,
		num_spatial_dims: int,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		*,
		activation_fn: Callable = nnx.relu,
		step_size: float = 1.0,
		cell_dropout_rate: float = 0.0,
		zeros_init: bool = False,
		rngs: nnx.Rngs,
	):
		"""Initialize the ResidualUpdate module.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			channel_size: Number of channels in the state.
			perception_size: Size of the perception input.
			hidden_layer_sizes: Sizes of hidden layers in the MLP.
			activation_fn: Activation function to use.
			step_size: Step size for the residual update.
			cell_dropout_rate: Dropout rate for cell updates.
			zeros_init: Whether to use zeros initialization for the weights of the last layer.
			rngs: rng key.

		"""
		super().__init__(
			num_spatial_dims=num_spatial_dims,
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			activation_fn=activation_fn,
			zeros_init=zeros_init,
			rngs=rngs,
		)
		self.dropout = nnx.Dropout(rate=cell_dropout_rate, broadcast_dims=(-1,), rngs=rngs)
		self.step_size = step_size

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		Args:
			state: Current state.
			perception: Current perception.
			input: Optional input.

		Returns:
			Next state.

		"""
		update = super().__call__(state, perception, input)
		update = self.dropout(update)
		state += self.step_size * update
		return state
