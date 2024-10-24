"""Convolution Perceive module."""

from collections.abc import Callable

from flax import nnx

from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State


class ConvPerceive(Perceive):
	"""Convolution Perceive class."""

	conv: nnx.Conv
	activation_fn: Callable | None

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		rngs: nnx.Rngs,
		*,
		kernel_size: int | tuple[int, ...] = (3, 3),
		padding: str = "SAME",
		feature_group_count: int = 1,
		use_bias: bool = False,
		activation_fn: Callable | None = None,
	):
		"""Initialize the ConvPerceive layer.

		Args:
			channel_size: Number of input channels.
			perception_size: Number of output perception features.
			rngs: Random number generators.
			kernel_size: Size of the convolutional kernel.
			padding: Padding to use.
			feature_group_count: Number of feature groups.
			use_bias: Whether to use bias in convolutional layers.
			activation_fn: Activation function to use.

		"""
		self.conv = nnx.Conv(
			in_features=channel_size,
			out_features=perception_size,
			kernel_size=kernel_size,
			padding=padding,
			feature_group_count=feature_group_count,
			use_bias=use_bias,
			rngs=rngs,
		)
		self.activation_fn = activation_fn

	def __call__(self, state: State) -> Perception:
		"""Apply perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		perception = self.conv(state)
		return self.activation_fn(perception) if self.activation_fn else perception
