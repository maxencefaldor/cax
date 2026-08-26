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
		attraction_matrix = jax.random.uniform(key, (num_classes, num_classes))
		particle_life = ParticleLife(
			num_classes=num_classes,
			dt=0.01,
			attraction_matrix=attraction_matrix,
		)
		return particle_life

	try:
		init_particle_life()
	except Exception as e:
		pytest.fail(f"ParticleLife instantiation failed under jit: {e}")


def test_particle_life_gradient_is_finite() -> None:
	"""Test that gradients through the perception are finite despite the zero diagonal."""
	import jax.numpy as jnp

	from cax.cs.particle_life import ParticleLifePerceive, ParticleLifeState

	attraction_matrix = jnp.array([[1.0, -0.5], [0.5, 1.0]])
	perceive = ParticleLifePerceive(attraction_matrix=attraction_matrix)

	key = jax.random.key(0)
	position = jax.random.uniform(key, (16, 2))
	class_id = jnp.zeros((16,), dtype=jnp.int32).at[8:].set(1)
	velocity = jnp.zeros((16, 2))

	def loss(position: jax.Array) -> jax.Array:
		state = ParticleLifeState(class_id=class_id, position=position, velocity=velocity)
		return jnp.sum(perceive(state).acceleration)

	grad = jax.grad(loss)(position)
	assert jnp.isfinite(grad).all()
