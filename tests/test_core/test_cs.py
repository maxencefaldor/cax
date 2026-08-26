"""Tests for the ComplexSystem multi-step driver."""

import jax.numpy as jnp
import pytest
from flax import nnx
from jax import Array

from cax.cs.elementary import Elementary


@pytest.fixture
def elementary() -> Elementary:
	"""Return a rule 110 Elementary Cellular Automaton."""
	wolfram_code = Elementary.wolfram_code_from_rule_number(110)
	return Elementary(wolfram_code=wolfram_code, rngs=nnx.Rngs(0))


def state_init(width: int = 32) -> Array:
	"""Return a single-seed initial state."""
	return jnp.zeros((width, 1)).at[width // 2].set(1.0)


def test_trajectory_matches_final_state(elementary: Elementary) -> None:
	"""Test that the trajectory stacks per-step states and ends at the final state."""
	num_steps = 8
	state_final, states = elementary(state_init(), num_steps=num_steps, trajectory=True)

	assert states.shape == (num_steps, 32, 1)
	assert jnp.array_equal(states[-1], state_final)

	state_one = elementary(state_init(), num_steps=1)
	assert jnp.array_equal(states[0], state_one)


def test_trajectory_leaves_no_module_state(elementary: Elementary) -> None:
	"""Test that requesting a trajectory leaves nothing behind on the module.

	The scan returns the trajectory as stacked outputs, so consecutive calls with
	different step counts must be independent — no stale intermediates.
	"""
	elementary(state_init(), num_steps=5, trajectory=True)
	_, states = elementary(state_init(), num_steps=7, trajectory=True)
	assert states.shape[0] == 7

	intermediates = nnx.state(elementary, nnx.Intermediate)
	assert not nnx.to_flat_state(intermediates)


def test_final_only_call_matches_trajectory_call(elementary: Elementary) -> None:
	"""Test that trajectory=False returns the same final state as trajectory=True."""
	state_final = elementary(state_init(), num_steps=6)
	state_final_trajectory, _ = elementary(state_init(), num_steps=6, trajectory=True)
	assert jnp.array_equal(state_final, state_final_trajectory)


def test_vmap_over_shared_module(elementary: Elementary) -> None:
	"""Test that a batch of states rolls out under one shared module via nnx.vmap."""
	import jax

	batch = jax.random.bernoulli(jax.random.key(0), 0.5, (4, 32, 1)).astype(jnp.float32)
	batch_final = nnx.vmap(lambda cs, state: cs(state, num_steps=3), in_axes=(None, 0))(
		elementary, batch
	)
	assert batch_final.shape == (4, 32, 1)

	state_final = elementary(batch[0], num_steps=3)
	assert jnp.array_equal(batch_final[0], state_final)
