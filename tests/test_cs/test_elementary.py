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
