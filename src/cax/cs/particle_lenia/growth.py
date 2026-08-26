"""Particle Lenia growth module.

References:
	[1] Particle Lenia and the energy-based formulation, Mordvintsev et al. 2022.
		https://google-research.github.io/self-organising-systems/particle-lenia/

"""

from flax import nnx
from jax import Array

from .kernel import bell


@nnx.dataclass
class ParticleLeniaGrowthParams(nnx.Pytree):
	"""Growth parameters."""

	mean: Array = nnx.data()
	std: Array = nnx.data()


def peak_growth_fn(u: Array, growth_params: ParticleLeniaGrowthParams) -> Array:
	"""Growth function introduced in [1].

	The reference defines growth as `G = peak_f(U, mu_g, sigma_g)` — the same Gaussian
	bump as the kernel, without the 1/2 factor and without the affine `2·(…) - 1` of
	grid Lenia's growth mapping. Using grid Lenia's growth here would widen the bump by
	a factor of sqrt(2) and double attraction relative to repulsion, so published
	Particle Lenia parameters would not reproduce.

	Args:
		u: Kernel field values.
		growth_params: Growth parameters (mean and std of the bump).

	Returns:
		Growth field values.

	"""
	return bell(u, growth_params.mean, growth_params.std)
