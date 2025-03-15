"""Particle Lenia model.

[1] https://google-research.github.io/self-organising-systems/particle-lenia/
"""

from collections.abc import Callable

from cax.core.ca import CA

from .particle_lenia_perceive import ParticleLeniaPerceive, peak_kernel_fn
from .particle_lenia_update import ParticleLeniaUpdate
from .types import RuleParams


class ParticleLenia(CA):
	"""Particle Lenia model."""

	def __init__(
		self,
		num_dims: int,
		T: int,
		rule_params: RuleParams,
		kernel_fn: Callable = peak_kernel_fn,
		growth_fn: Callable = peak_kernel_fn,
	):
		"""Initialize Particle Lenia."""
		perceive = ParticleLeniaPerceive(
			num_dims=num_dims,
			rule_params=rule_params,
			kernel_fn=kernel_fn,
		)
		update = ParticleLeniaUpdate(
			T=T,
		)

		super().__init__(perceive, update)
