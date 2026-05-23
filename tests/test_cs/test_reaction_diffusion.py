"""Tests for Reaction-Diffusion (Gray-Scott)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cax.cs.reaction_diffusion import ReactionDiffusion


def test_reaction_diffusion_jit_init() -> None:
	"""Test that ReactionDiffusion can be instantiated under jax.jit."""

	@jax.jit
	def init_reaction_diffusion() -> ReactionDiffusion:
		rngs = nnx.Rngs(0)
		cs = ReactionDiffusion(rngs=rngs)
		return cs

	try:
		init_reaction_diffusion()
	except Exception as e:
		pytest.fail(f"ReactionDiffusion instantiation failed under jit: {e}")


def test_reaction_diffusion_step() -> None:
	"""Test that a single step produces a valid state."""
	rngs = nnx.Rngs(0)
	cs = ReactionDiffusion(rngs=rngs)

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2))
	state = state.at[14:18, 14:18, 1].set(0.5)

	next_state = cs._step(state)

	assert next_state.shape == state.shape
	assert jnp.all(next_state >= 0.0)
	assert jnp.all(next_state <= 1.0)


def test_reaction_diffusion_multi_step() -> None:
	"""Test that multi-step evolution works."""
	rngs = nnx.Rngs(0)
	cs = ReactionDiffusion(rngs=rngs)

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2))
	state = state.at[14:18, 14:18, 1].set(0.5)

	final_state = cs(state, num_steps=10)

	assert final_state.shape == state.shape
	assert jnp.all(final_state >= 0.0)
	assert jnp.all(final_state <= 1.0)


def test_reaction_diffusion_render() -> None:
	"""Test that render produces a valid RGB uint8 image."""
	rngs = nnx.Rngs(0)
	cs = ReactionDiffusion(rngs=rngs)

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2)) * 0.5

	rgb = cs.render(state)

	assert rgb.shape == (*spatial_dims, 3)
	assert rgb.dtype == jnp.uint8
