"""Tests for Particle Lenia."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.particle_lenia import ParticleLenia, ParticleLeniaRuleParams
from cax.cs.particle_lenia.growth import GrowthParams
from cax.cs.particle_lenia.kernel import KernelParams


def test_particle_lenia_jit_init() -> None:
	"""Test that ParticleLenia can be instantiated under jax.jit."""

	@jax.jit
	def init_particle_lenia() -> ParticleLenia:
		kernel_params = KernelParams(
			weight=jnp.array([1.0]), mean=jnp.array([0.5]), std=jnp.array([0.1])
		)
		growth_params = GrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
		rule_params = ParticleLeniaRuleParams(
			c_rep=1.0,
			kernel_params=kernel_params,
			growth_params=growth_params,
		)
		particle_lenia = ParticleLenia(
			num_spatial_dims=2,
			T=10,
			rule_params=rule_params,
		)
		return particle_lenia

	try:
		init_particle_lenia()
	except Exception as e:
		pytest.fail(f"ParticleLenia instantiation failed under jit: {e}")
