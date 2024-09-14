"""Depthwise Convolution Perceive module."""

from collections.abc import Sequence

from flax import nnx

from cax.core.perceive.perceive import Perceive
from cax.types import Perception, State


class DWConvPerceive(Perceive):
	"""Depthwise Convolution Perceive class."""

	dwconv: nnx.Conv

	def __init__(
		self,
		channel_size: int,
		rngs: nnx.Rngs,
		*,
		num_kernels: int = 3,
		kernel_size: Sequence[int] = (3, 3),
		use_bias: bool = False,
	):
		"""Initialize the DWConvPerceive module.

		Args:
			channel_size: Number of input channels.
			rngs: Random number generator states.
			num_kernels: Number of kernels to use.
			kernel_size: Size of the convolution kernel.
			use_bias: Whether to use bias in the convolution.

		"""
		self.dwconv = nnx.Conv(
			channel_size,
			num_kernels * channel_size,
			kernel_size=kernel_size,
			feature_group_count=channel_size,
			use_bias=use_bias,
			rngs=rngs,
		)

	def __call__(self, state: State) -> Perception:
		"""Apply depthwise convolution to the input state.

		Args:
			state: Input state to be processed.

		Returns:
			The processed perception.

		"""
		perception = self.dwconv(state)
		return perception
