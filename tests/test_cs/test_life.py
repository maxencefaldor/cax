"""Tests for Life."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.life import Life


def test_life_jit_init() -> None:
	"""Test that Life can be instantiated under jax.jit."""

	@jax.jit
	def init_life() -> Life:
		birth = jnp.zeros(9)
		survival = jnp.zeros(9)
		life = Life(birth=birth, survival=survival)
		return life

	try:
		init_life()
	except Exception as e:
		pytest.fail(f"Life instantiation failed under jit: {e}")


def test_blinker_oscillates_with_period_two() -> None:
	"""Test the blinker: a period-2 oscillator under B3/S23."""
	birth, survival = Life.birth_survival_from_string("B3/S23")
	life = Life(birth=birth, survival=survival)

	state = jnp.zeros((8, 8, 1)).at[4, 3:6, 0].set(1.0)

	state_one = life(state, num_steps=1)
	state_two = life(state, num_steps=2)

	vertical = jnp.zeros((8, 8, 1)).at[3:6, 4, 0].set(1.0)
	assert jnp.array_equal(state_one, vertical)
	assert jnp.array_equal(state_two, state)
