"""Tests for Sandpile (Abelian Sandpile)."""

import jax
import jax.numpy as jnp
import pytest

from cax.cs.sandpile import Sandpile


def test_sandpile_jit_init() -> None:
	"""Test that Sandpile can be instantiated under jax.jit."""

	@jax.jit
	def init_sandpile() -> Sandpile:
		cs = Sandpile()
		return cs

	try:
		init_sandpile()
	except Exception as e:
		pytest.fail(f"Sandpile instantiation failed under jit: {e}")


def test_sandpile_toppling() -> None:
	"""Test that a single step topples critical cells correctly."""
	cs = Sandpile()

	spatial_dims = (5, 5)
	state = jnp.zeros((*spatial_dims, 1))
	state = state.at[2, 2, 0].set(8.0)

	next_state = cs._step(state)

	assert next_state.shape == state.shape
	assert next_state[2, 2, 0] < 8.0


def test_sandpile_mass_conservation() -> None:
	"""Test that total mass is conserved with periodic boundaries."""
	cs = Sandpile()

	spatial_dims = (8, 8)
	state = jnp.zeros((*spatial_dims, 1))
	state = state.at[4, 4, 0].set(16.0)

	total_before = jnp.sum(state)
	next_state = cs._step(state)
	total_after = jnp.sum(next_state)

	assert jnp.allclose(total_before, total_after)


def test_sandpile_stable_state_unchanged() -> None:
	"""Test that a stable configuration (all cells < threshold) is unchanged."""
	cs = Sandpile()

	spatial_dims = (8, 8)
	state = jnp.ones((*spatial_dims, 1)) * 3.0

	next_state = cs._step(state)

	assert jnp.allclose(state, next_state)


def test_sandpile_multi_step() -> None:
	"""Test that multi-step evolution works."""
	cs = Sandpile()

	spatial_dims = (8, 8)
	state = jnp.zeros((*spatial_dims, 1))
	state = state.at[4, 4, 0].set(64.0)

	final_state = cs(state, num_steps=20)

	assert final_state.shape == state.shape
	assert jnp.allclose(jnp.sum(state), jnp.sum(final_state))


def test_sandpile_render() -> None:
	"""Test that render produces a valid RGB uint8 image."""
	cs = Sandpile()

	spatial_dims = (8, 8)
	state = jnp.ones((*spatial_dims, 1)) * 2.0

	rgb = cs.render(state)

	assert rgb.shape == (*spatial_dims, 3)
	assert rgb.dtype == jnp.uint8
