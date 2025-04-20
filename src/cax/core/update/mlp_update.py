"""Multi-Layer Perceptron update module."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import default_kernel_init

from cax.core.update.update import Update
from cax.types import Input, Perception, State


class MLPUpdate(Update):
	"""Multi-Layer Perceptron update class."""

	def __init__(
		self,
		num_spatial_dims: int,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		rngs: nnx.Rngs,
		*,
		activation_fn: Callable = nnx.relu,
		zeros_init: bool = False,
	):
		"""Initialize the MLPUpdate layer.

		Args:
			num_spatial_dims: Number of spatial dimensions.
			channel_size: Number of channels in the output.
			perception_size: Size of the input perception.
			hidden_layer_sizes: Sizes of hidden layers.
			rngs: Random number generators.
			activation_fn: Activation function to use.
			zeros_init: Whether to use zeros initialization for the weights of the last layer.

		"""
		in_features = (perception_size,) + hidden_layer_sizes
		out_features = hidden_layer_sizes + (channel_size,)
		kernel_init = [default_kernel_init for _ in hidden_layer_sizes] + [
			initializers.zeros_init() if zeros_init else default_kernel_init
		]
		self.layers = [
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
		self.activation_fn = activation_fn

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the MLP update to the given state and perception.

		If an input is provided, it is concatenated to the perception
		before being passed through the layers of the MLP.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state from the previous step.
			input: Optional external input to the system.

		Returns:
			Updated state after applying the MLP layers.

		"""
		if input is not None:
			perception = jnp.concatenate([perception, input], axis=-1)

		for layer in self.layers[:-1]:
			perception = self.activation_fn(layer(perception))
		state = self.layers[-1](perception)
		return state
