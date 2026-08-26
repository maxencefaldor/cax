"""Tests for Boids."""

import jax
import pytest
from flax import nnx

from cax.cs.boids import Boids, BoidsPolicy


def test_boids_jit_init() -> None:
	"""Test that Boids can be instantiated under jax.jit."""

	@jax.jit
	def init_boids() -> Boids:
		rngs = nnx.Rngs(0)
		policy = BoidsPolicy(rngs=rngs)
		boids = Boids(dt=0.01, velocity_half_life=0.1, boid_policy=policy)
		return boids

	try:
		init_boids()
	except Exception as e:
		pytest.fail(f"Boids instantiation failed under jit: {e}")


def test_boid_policy_isolated_boid_is_finite() -> None:
	"""Test that a boid with no neighbor in range produces finite forces and gradients."""
	import jax.numpy as jnp
	from flax import nnx

	from cax.cs.boids import BoidsPolicy, BoidsState

	policy = BoidsPolicy(perception_radius=0.01, noise_scale=0.0, rngs=nnx.Rngs(0))

	# Two boids far apart on the torus: both are isolated.
	position = jnp.array([[0.1, 0.1], [0.6, 0.6]])
	velocity = jnp.zeros((2, 2))

	acceleration = policy(BoidsState(position=position, velocity=velocity), 0)
	assert jnp.isfinite(acceleration).all()

	# The policy is stateful (it draws rng noise), so for a raw jax transform it is
	# constructed inside the traced function: all of its state then lives at the
	# inner trace level.
	def loss(position: jax.Array) -> jax.Array:
		inner_policy = BoidsPolicy(perception_radius=0.01, noise_scale=0.0, rngs=nnx.Rngs(0))
		return jnp.sum(inner_policy(BoidsState(position=position, velocity=velocity), 0))

	grad = jax.grad(loss)(position)
	assert jnp.isfinite(grad).all()
