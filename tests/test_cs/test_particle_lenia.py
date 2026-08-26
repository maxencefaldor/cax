"""Tests for Particle Lenia."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.particle_lenia import ParticleLenia, ParticleLeniaRuleParams, ParticleLeniaState
from cax.cs.particle_lenia.growth import ParticleLeniaGrowthParams
from cax.cs.particle_lenia.kernel import ParticleLeniaKernelParams


def test_particle_lenia_jit_init() -> None:
	"""Test that ParticleLenia can be instantiated under jax.jit."""

	@jax.jit
	def init_particle_lenia() -> ParticleLenia:
		kernel_params = ParticleLeniaKernelParams(
			weight=jnp.array([1.0]), mean=jnp.array([0.5]), std=jnp.array([0.1])
		)
		growth_params = ParticleLeniaGrowthParams(mean=jnp.array([0.5]), std=jnp.array([0.1]))
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


def test_particle_lenia_fields_match_reference() -> None:
	"""Test compute_fields against the reference formulas of Mordvintsev et al."""
	from cax.cs.particle_lenia import ParticleLeniaPerceive

	# The reference notebook's parameters.
	mu_k, sigma_k, w_k = 4.0, 1.0, 0.022
	mu_g, sigma_g, c_rep = 0.6, 0.15, 1.0

	kernel_params = ParticleLeniaKernelParams(
		weight=jnp.array(w_k), mean=jnp.array(mu_k), std=jnp.array(sigma_k)
	)
	growth_params = ParticleLeniaGrowthParams(mean=jnp.array(mu_g), std=jnp.array(sigma_g))
	rule_params = ParticleLeniaRuleParams(
		c_rep=c_rep, kernel_params=kernel_params, growth_params=growth_params
	)
	perceive = ParticleLeniaPerceive(num_spatial_dims=2, rule_params=rule_params)

	key = jax.random.key(0)
	state = ParticleLeniaState(position=jax.random.uniform(key, (12, 2), minval=-4.0, maxval=4.0))
	x = jnp.array([0.5, -0.25])

	U, G, R = perceive.compute_fields(state, x)

	# Reference: U = sum_i w_k * exp(-((r_i - mu_k) / sigma_k)^2);
	# G = exp(-((U - mu_g) / sigma_g)^2); R = c_rep / 2 * sum_i max(1 - r_i, 0)^2.
	r = jnp.sqrt(jnp.sum(jnp.square(x - state.position), axis=-1))
	U_reference = jnp.sum(w_k * jnp.exp(-(((r - mu_k) / sigma_k) ** 2)))
	G_reference = jnp.exp(-(((U_reference - mu_g) / sigma_g) ** 2))
	R_reference = 0.5 * c_rep * jnp.sum(jnp.maximum(1.0 - r, 0.0) ** 2)

	assert jnp.allclose(U, U_reference, atol=1e-6)
	assert jnp.allclose(G, G_reference, atol=1e-6)
	assert jnp.allclose(R, R_reference, atol=1e-6)


def test_particle_lenia_render_modes_differ() -> None:
	"""Test that the three render modes run and produce distinct images."""
	kernel_params = ParticleLeniaKernelParams(
		weight=jnp.array(0.022), mean=jnp.array(4.0), std=jnp.array(1.0)
	)
	growth_params = ParticleLeniaGrowthParams(mean=jnp.array(0.6), std=jnp.array(0.15))
	rule_params = ParticleLeniaRuleParams(
		c_rep=1.0, kernel_params=kernel_params, growth_params=growth_params
	)
	particle_lenia = ParticleLenia(num_spatial_dims=2, T=10, rule_params=rule_params)

	key = jax.random.key(0)
	state = ParticleLeniaState(position=jax.random.uniform(key, (16, 2), minval=-6.0, maxval=6.0))

	images = {
		mode: particle_lenia.render(state, resolution=64, mode=mode)
		for mode in ("particles", "UG", "E")
	}
	for image in images.values():
		assert image.shape == (64, 64, 3)
	assert not jnp.array_equal(images["particles"], images["UG"])
	assert not jnp.array_equal(images["UG"], images["E"])
