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
		cs = ReactionDiffusion()
		return cs

	try:
		init_reaction_diffusion()
	except Exception as e:
		pytest.fail(f"ReactionDiffusion instantiation failed under jit: {e}")


def test_reaction_diffusion_step() -> None:
	"""Test that a single step produces a valid state."""
	cs = ReactionDiffusion()

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2))
	state = state.at[14:18, 14:18, 1].set(0.5)

	next_state = cs._step(state)

	assert next_state.shape == state.shape
	assert jnp.all(next_state >= 0.0)
	assert jnp.all(next_state <= 1.0)


def test_reaction_diffusion_multi_step() -> None:
	"""Test that multi-step evolution works."""
	cs = ReactionDiffusion()

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2))
	state = state.at[14:18, 14:18, 1].set(0.5)

	final_state = cs(state, num_steps=10)

	assert final_state.shape == state.shape
	assert jnp.all(final_state >= 0.0)
	assert jnp.all(final_state <= 1.0)


def test_reaction_diffusion_render() -> None:
	"""Test that render produces a valid RGB uint8 image."""
	cs = ReactionDiffusion()

	spatial_dims = (32, 32)
	state = jnp.ones((*spatial_dims, 2)) * 0.5

	rgb = cs.render(state)

	assert rgb.shape == (*spatial_dims, 3)
	assert rgb.dtype == jnp.uint8


def test_reaction_diffusion_perception_is_identity_and_laplacian() -> None:
	"""Test the perception channels against a manual periodic Laplacian."""
	from cax.cs.reaction_diffusion import ReactionDiffusionPerceive

	perceive = ReactionDiffusionPerceive()
	key = jax.random.key(0)
	state = jax.random.uniform(key, (8, 8, 2))

	perception = perceive(state)
	assert perception.shape == (8, 8, 4)

	def laplacian(field: jax.Array) -> jax.Array:
		return (
			jnp.roll(field, 1, axis=0)
			+ jnp.roll(field, -1, axis=0)
			+ jnp.roll(field, 1, axis=1)
			+ jnp.roll(field, -1, axis=1)
			- 4.0 * field
		)

	assert jnp.allclose(perception[..., 0], state[..., 0], atol=1e-6)
	assert jnp.allclose(perception[..., 1], laplacian(state[..., 0]), atol=1e-5)
	assert jnp.allclose(perception[..., 2], state[..., 1], atol=1e-6)
	assert jnp.allclose(perception[..., 3], laplacian(state[..., 1]), atol=1e-5)


def test_reaction_diffusion_stencil_is_not_trainable() -> None:
	"""Test that the physics stencil never appears among the trainable parameters."""
	from cax.cs.reaction_diffusion import ReactionDiffusion

	reaction_diffusion = ReactionDiffusion()
	params = nnx.state(reaction_diffusion, nnx.Param)
	assert not nnx.to_flat_state(params)


def test_gray_scott_state_stays_in_unit_interval() -> None:
	"""Test that concentrations remain in [0, 1] over many steps."""
	from cax.cs.reaction_diffusion import ReactionDiffusion

	reaction_diffusion = ReactionDiffusion()
	key = jax.random.key(0)
	state = jax.random.uniform(key, (32, 32, 2))

	state_final = reaction_diffusion(state, num_steps=200)
	assert jnp.min(state_final) >= 0.0
	assert jnp.max(state_final) <= 1.0
