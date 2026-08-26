"""Tests for the ComplexSystem multi-step driver."""

import warnings

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


def test_rollout_matches_final_state(elementary: Elementary) -> None:
	"""Test that the rollout stacks per-step states and ends at the final state."""
	num_steps = 8
	state_final, states = elementary.rollout(state_init(), num_steps=num_steps)

	assert states.shape == (num_steps, 32, 1)
	assert jnp.array_equal(states[-1], state_final)

	state_one = elementary(state_init(), num_steps=1)
	assert jnp.array_equal(states[0], state_one)


def test_rollout_leaves_no_module_state(elementary: Elementary) -> None:
	"""Test that a rollout leaves nothing behind on the module.

	The trajectory is a scan output, so consecutive rollouts with different step
	counts must be independent — no stale intermediates.
	"""
	elementary.rollout(state_init(), num_steps=5)
	_, states = elementary.rollout(state_init(), num_steps=7)
	assert states.shape[0] == 7

	intermediates = nnx.state(elementary, nnx.Intermediate)
	assert not nnx.to_flat_state(intermediates)


def test_driver_emits_no_warning(elementary: Elementary) -> None:
	"""Test that neither driver hits a deprecation path (e.g. sow outside nnx.capture)."""
	with warnings.catch_warnings():
		warnings.simplefilter("error", DeprecationWarning)
		elementary(state_init(), num_steps=4)
		elementary.rollout(state_init(), num_steps=4)


def test_call_matches_rollout_final_state(elementary: Elementary) -> None:
	"""Test that `__call__` returns the same final state as `rollout`."""
	state_final = elementary(state_init(), num_steps=6)
	state_final_rollout, _ = elementary.rollout(state_init(), num_steps=6)
	assert jnp.array_equal(state_final, state_final_rollout)


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
