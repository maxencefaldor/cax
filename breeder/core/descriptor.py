"""Descriptor module.

A descriptor is a weightless reduction of named phenotype series to a behavioral
descriptor vector: each axis group is the windowed time-mean of one series, divided by a
scale that makes axes commensurate. Scalar series contribute one axis (the official
hand-crafted descriptors); vector series contribute their full width (mean latent =
AURORA, mean VGG features = pretrained perceptual descriptor).
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from .phenotype import Phenotype


class Descriptor(nnx.Module):
	"""Reduction of named phenotype series to a descriptor vector."""

	def __init__(self, series: tuple[tuple[str, float], ...], *, window: int):
		"""Initialize the descriptor.

		Args:
			series: Pairs of (series name, scale); each series' windowed time-mean,
				divided by its scale, contributes its width to the descriptor.
			window: Number of final steps to reduce over, excluding the developmental
				transient.

		"""
		self.series = series
		self.window = window

	def __call__(self, phenotype: Phenotype) -> Array:
		"""Compute the descriptor of a phenotype."""
		return self.reduce(phenotype.series)

	def reduce(self, series: dict[str, Array]) -> Array:
		"""Reduce named series to the descriptor vector.

		Exposed separately so the population can be re-encoded from archived
		observations after an encoder refit.
		"""
		return jnp.concatenate(
			[
				jnp.atleast_1d(jnp.mean(series[name][-self.window :], axis=0)) / scale
				for name, scale in self.series
			]
		)


class DescriptorConfig(BaseModel):
	"""Config of `Descriptor`; series references are validated against the experiment.

	The default scales of the hand-crafted axes are rough Lenia magnitudes
	(mass ~10, linear_velocity ~0.01); encoder series use scale 1.0.
	"""

	model_config = ConfigDict(frozen=True, extra="forbid")

	series: tuple[tuple[str, PositiveFloat], ...] = (("latent", 1.0),)
	window: PositiveInt = 32

	def build(self) -> Descriptor:
		"""Build the configured descriptor."""
		return Descriptor(self.series, window=self.window)
