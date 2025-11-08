"""MLP update module."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import default_kernel_init

from cax.core import Input, State
from cax.core.perceive import Perception

from .update import Update


class MLPUpdate(Update):
	"""MLP update class.

	Maps a perception (and optional input) to the next state using pointwise convolutional
	layers (kernel size 1) applied independently at each spatial position.

	"""

	def __init__(
		self,
		num_spatial_dims: int,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		*,
		activation_fn: Callable = nnx.relu,
		zeros_init: bool = False,
		rngs: nnx.Rngs,
	):
		"""Initialize MLP update.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			channel_size: Number of channels in the output.
			perception_size: Size of the input perception.
			hidden_layer_sizes: Sizes of hidden layers.
			activation_fn: Activation function to use.
			zeros_init: Whether to use zeros initialization for the weights of the last layer.
			rngs: rng key.

		"""
		in_features = (perception_size,) + hidden_layer_sizes
		out_features = hidden_layer_sizes + (channel_size,)
		kernel_init = [default_kernel_init for _ in hidden_layer_sizes] + [
			initializers.zeros_init() if zeros_init else default_kernel_init
		]
		self.layers = nnx.List(
			[
				nnx.Conv(
					in_features,
					out_features,
					kernel_size=num_spatial_dims * (1,),
					kernel_init=kernel_init,
					rngs=rngs,
				)
				for in_features, out_features, kernel_init in zip(
					in_features, out_features, kernel_init
				)
			]
		)
		self.activation_fn = activation_fn

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.

		If input is provided, it is concatenated to the perception along the channel axis
		before being passed through the layers.

		Args:
			state: Current state.
			perception: Current perception.
			input: Optional input.

		Returns:
			Next state.

		"""
		if input is not None:
			perception = jnp.concatenate([perception, input], axis=-1)

		for layer in self.layers[:-1]:
			perception = self.activation_fn(layer(perception))
		state = self.layers[-1](perception)
		return state
