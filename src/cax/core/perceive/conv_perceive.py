"""Convolution perceive module."""

from collections.abc import Callable

from flax import nnx

from cax.core import State

from .perceive import Perceive, Perception


class ConvPerceive(Perceive):
	"""Convolution perceive class."""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		*,
		kernel_size: int | tuple[int, ...] = (3, 3),
		padding: str = "SAME",
		feature_group_count: int = 1,
		use_bias: bool = False,
		activation_fn: Callable | None = None,
		rngs: nnx.Rngs,
	):
		"""Initialize convolution perceive.

		Args:
			channel_size: Number of input channels.
			perception_size: Number of output perception features.
			kernel_size: Size of the convolutional kernel.
			padding: Padding to use.
			feature_group_count: Number of feature groups.
			use_bias: Whether to use bias in convolutional layers.
			activation_fn: Activation function to use.
			rngs: rng key.

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
		"""Apply convolutional perception to the input state.

		Inputs are expected to have shape `(..., *spatial_dims, channel_size)` where `spatial_dims`
		is a tuple of `num_spatial_dims` dimensions and `channel_size` is the number of channels.
		The output shape is `(..., *spatial_dims, perception_size)`. If `activation_fn` is provided,
		it is applied element-wise to the convolution output. If `activation_fn` is `None`, the
		convolution output is returned as is.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		perception = self.conv(state)
		return self.activation_fn(perception) if self.activation_fn else perception
