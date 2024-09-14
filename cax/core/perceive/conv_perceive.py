"""Convolution Perceive module."""

from collections.abc import Callable

from flax import nnx

from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State


class ConvPerceive(Perceive):
	"""Convolution Perceive class."""

	layers: list[nnx.Conv]
	activation_fn: Callable

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...],
		rngs: nnx.Rngs,
		*,
		kernel_size: int | tuple[int, ...] = (3, 3),
		use_bias: bool = False,
		activation_fn: Callable = nnx.relu,
	):
		"""Initialize the ConvPerceive layer.

		Args:
			channel_size: Number of input channels.
			perception_size: Size of the output perception.
			hidden_layer_sizes: Sizes of hidden layers.
			rngs: Random number generators.
			kernel_size: Size of the convolutional kernel.
			use_bias: Whether to use bias in convolutional layers.
			activation_fn: Activation function to use.

		"""
		in_features = (channel_size,) + hidden_layer_sizes
		out_features = hidden_layer_sizes + (perception_size,)
		self.layers = [
			nnx.Conv(in_features, out_features, kernel_size=kernel_size, use_bias=use_bias, rngs=rngs)
			for in_features, out_features in zip(in_features, out_features)
		]
		self.activation_fn = activation_fn

	def __call__(self, state: State) -> Perception:
		"""Apply perception to the input state.

		Args:
			state: Input state of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		perception = state
		for layer in self.layers[:-1]:
			perception = self.activation_fn(layer(perception))
		perception = self.layers[-1](perception)
		return perception
