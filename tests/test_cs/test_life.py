"""Tests for Life."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cax.cs.life import Life


def test_life_jit_init() -> None:
	"""Test that Life can be instantiated under jax.jit."""

	@jax.jit
	def init_life() -> Life:
		rngs = nnx.Rngs(0)
		birth = jnp.zeros(9)
		survival = jnp.zeros(9)
		life = Life(birth=birth, survival=survival, rngs=rngs)
		return life

	try:
		init_life()
	except Exception as e:
		pytest.fail(f"Life instantiation failed under jit: {e}")
