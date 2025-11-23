"""Tests for Boids."""

import jax
import pytest
from flax import nnx

from cax.cs.boids import BoidPolicy, Boids


def test_boids_jit_init() -> None:
	"""Test that Boids can be instantiated under jax.jit."""

	@jax.jit
	def init_boids() -> Boids:
		rngs = nnx.Rngs(0)
		policy = BoidPolicy(rngs=rngs)
		boids = Boids(dt=0.01, velocity_half_life=0.1, boid_policy=policy)
		return boids

	try:
		init_boids()
	except Exception as e:
		pytest.fail(f"Boids instantiation failed under jit: {e}")
