"""Tests for Elementary Cellular Automata."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cax.cs.elementary import Elementary


def test_elementary_jit_init() -> None:
	"""Test that Elementary can be instantiated under jax.jit."""

	@jax.jit
	def init_elementary() -> Elementary:
		rngs = nnx.Rngs(0)
		wolfram_code = jnp.zeros(8)
		elementary = Elementary(wolfram_code=wolfram_code, rngs=rngs)
		return elementary

	try:
		init_elementary()
	except Exception as e:
		pytest.fail(f"Elementary instantiation failed under jit: {e}")


def test_rule_110_known_rows() -> None:
	"""Test rule 110 against hand-computed rows from a single seed."""
	from flax import nnx

	elementary = Elementary(
		wolfram_code=Elementary.wolfram_code_from_rule_number(110), rngs=nnx.Rngs(0)
	)
	state = jnp.zeros((8, 1)).at[5].set(1.0)

	_, states = elementary.rollout(state, num_steps=2)

	# Rule 110 from ...00000100...: one step grows left (00001100), two steps (00011100).
	assert jnp.array_equal(states[0][:, 0], jnp.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=jnp.float32))
	assert jnp.array_equal(states[1][:, 0], jnp.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=jnp.float32))
