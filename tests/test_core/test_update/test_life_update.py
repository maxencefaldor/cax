"""Tests for the LifeUpdate class."""

import jax.numpy as jnp
import pytest
from cax.core.ca import CA
from cax.core.perceive.depthwise_conv_perceive import DepthwiseConvPerceive
from cax.core.perceive.kernels import identity_kernel, neighbors_kernel
from cax.core.update.life_update import LifeUpdate
from flax import nnx


@pytest.fixture
def life_update():
	"""Fixture to create a LifeUpdate instance."""
	return LifeUpdate()


def test_life_update_initialization():
	"""Test the initialization of LifeUpdate."""
	update = LifeUpdate()
	assert isinstance(update, LifeUpdate)


@pytest.mark.parametrize(
	"state, perception, expected",
	[
		(jnp.array([[[0]]]), jnp.array([[[0, 3]]]), jnp.array([[[1]]])),  # Dead cell with 3 neighbors becomes alive
		(jnp.array([[[1]]]), jnp.array([[[1, 2]]]), jnp.array([[[1]]])),  # Live cell with 2 neighbors survives
		(jnp.array([[[1]]]), jnp.array([[[1, 1]]]), jnp.array([[[0]]])),  # Live cell with 1 neighbor dies
		(jnp.array([[[1]]]), jnp.array([[[1, 4]]]), jnp.array([[[0]]])),  # Live cell with 4 neighbors dies
		(jnp.array([[[0]]]), jnp.array([[[0, 2]]]), jnp.array([[[0]]])),  # Dead cell with 2 neighbors stays dead
	],
)
def test_life_update_rules(life_update, state, perception, expected):
	"""Test LifeUpdate with different Game of Life scenarios."""
	result = life_update(state, perception, None)
	assert jnp.array_equal(result, expected)


def test_life_update():
	"""Test the LifeUpdate in a CA simulation."""
	seed = 42
	spatial_dims = (16, 16)
	channel_size = 1
	num_steps = 4

	rngs = nnx.Rngs(seed)

	# Initialize state with a glider pattern
	state = jnp.zeros((*spatial_dims, channel_size))
	glider = jnp.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
	state = state.at[1:4, 1:4, 0].set(glider)

	perceive = DepthwiseConvPerceive(channel_size, rngs, num_kernels=2, kernel_size=(3, 3))
	update = LifeUpdate()

	# Set up perception kernels
	kernel = jnp.concatenate([identity_kernel(2), neighbors_kernel(2)], axis=-1)
	kernel = jnp.expand_dims(kernel, axis=-2)
	perceive.depthwise_conv.kernel = nnx.Param(kernel)

	ca = CA(perceive, update)

	state = ca(state, num_steps=num_steps)

	# Check if the glider has moved to the expected position after 4 steps
	assert jnp.array_equal(state[2:5, 2:5, 0], glider)
