"""Fitness module.

A fitness is a weightless reduction of one named phenotype series to a scalar, maximized
by the search. The series may come from the complex system's `develop` (e.g.
`norm_mean` of `velocity` selects for directed travel — net displacement per unit
time, which spinning cannot hack; mean `linear_velocity` is instantaneous speed,
directed or not) or from an experiment encoder (e.g. `-1 * var` of a
latent series is the paper's unsupervised homeostasis fitness; the mean of one channel of
a VGG feature series is a deepdream fitness).
"""

from typing import Annotated, Literal

import jax.numpy as jnp
from flax import nnx
from jax import Array
from pydantic import BaseModel, BeforeValidator, ConfigDict, NonNegativeInt, PositiveInt

from .phenotype import Phenotype

# Reductions apply over the time axis first. `var` is the *temporal* variance,
# averaged over any remaining dims (identical to jnp.var for scalar series): for a
# latent series, `-1 * var` is the official Leniabreeder unsupervised homeostasis
# fitness — a global variance would additionally penalize latent coordinates for
# differing from each other, i.e. reward looking like the VAE prior's mean.
# `norm_mean` is the norm of the time-mean: for the `velocity` vector series this is
# the net displacement per unit time — a spinner's rotating velocity vectors cancel,
# so it cannot hack the fitness the way instantaneous speed can
REDUCES = {
	"mean": jnp.mean,
	"var": lambda values: jnp.mean(jnp.var(values, axis=0)),
	"norm_mean": lambda values: jnp.linalg.norm(jnp.mean(values, axis=0)),
}


class Fitness(nnx.Module):
	"""Reduction of one named phenotype series to a scalar (higher is better)."""

	def __init__(
		self,
		series: str,
		reduce: Literal["mean", "var", "norm_mean"] = "mean",
		*,
		window: int,
		sign: float = 1.0,
		channel: int | None = None,
	):
		"""Initialize the fitness.

		Args:
			series: Name of the phenotype series to reduce.
			reduce: Reduction over the windowed series (over all remaining axes).
			window: Number of final steps to reduce over, excluding the developmental
				transient.
			sign: 1.0 to maximize the reduction, -1.0 to minimize it.
			channel: For vector series, the channel to reduce; None reduces all channels.

		"""
		self.series = series
		self.reduce = REDUCES[reduce]
		self.window = window
		self.sign = sign
		self.channel = channel

	def __call__(self, phenotype: Phenotype) -> Array:
		"""Compute the fitness of a phenotype."""
		values = phenotype.series[self.series][-self.window :]
		if self.channel is not None:
			values = values[:, self.channel]
		return self.sign * self.reduce(values)


class ReductionFitnessConfig(BaseModel):
	"""Literal fitness: a signed reduction over one named series.

	The config *is* the formula, in the code's own series vocabulary — the most
	rigorous specification for any objective whose definition is transparent.
	Series references are validated against the experiment.
	"""

	model_config = ConfigDict(frozen=True, extra="forbid")

	series: str = "velocity"
	reduce: Literal["mean", "var", "norm_mean"] = "norm_mean"
	# BeforeValidator(int): the CLI delivers "-1" as a string, which Literal won't coerce
	sign: Annotated[Literal[1, -1], BeforeValidator(int)] = 1
	channel: NonNegativeInt | None = None
	window: PositiveInt = 32

	def build(self) -> Fitness:
		"""Build the configured fitness."""
		return Fitness(
			self.series,
			self.reduce,
			window=self.window,
			sign=float(self.sign),
			channel=self.channel,
		)


class HomeostasisFitnessConfig(BaseModel):
	"""The official Leniabreeder unsupervised homeostasis fitness (arXiv:2406.04235).

	Maximizes the negative temporal variance of a learned feature series: for latents
	`z[t, d]` over the window, `f = -mean_d Var_t(z[:, d])` — stability of the encoded
	identity over time. Named (rather than spelled as a literal reduction) because its
	definition is paper-pinned and subtle: a *global* variance would additionally
	reward sitting at the VAE prior's mean. The battery asserts the built fitness
	against the hand-derived formula.
	"""

	model_config = ConfigDict(frozen=True, extra="forbid")

	name: Literal["homeostasis"] = "homeostasis"
	series: str = "latent"
	window: PositiveInt = 32

	def build(self) -> Fitness:
		"""Build the homeostasis fitness."""
		return Fitness(self.series, "var", window=self.window, sign=-1.0)


# Named objectives come first so their `name` tag decides; a literal reduction config
# (which forbids `name` as an extra field) is the fallback — no tag needed on it
FitnessConfig = HomeostasisFitnessConfig | ReductionFitnessConfig
