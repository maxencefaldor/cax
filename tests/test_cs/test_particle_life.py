"""Tests for Particle Life."""

import jax
import pytest

from cax.cs.particle_life import ParticleLife


def test_particle_life_jit_init() -> None:
	"""Test that ParticleLife can be instantiated under jax.jit."""

	@jax.jit
	def init_particle_life() -> ParticleLife:
		num_classes = 3
		key = jax.random.key(0)
		A = jax.random.uniform(key, (num_classes, num_classes))
		particle_life = ParticleLife(
			num_classes=num_classes,
			dt=0.01,
			A=A,
		)
		return particle_life

	try:
		init_particle_life()
	except Exception as e:
		pytest.fail(f"ParticleLife instantiation failed under jit: {e}")
