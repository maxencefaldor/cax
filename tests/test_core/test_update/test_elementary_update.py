"""Tests for the ElementaryUpdate class."""

import jax.numpy as jnp
import pytest
from cax.core.ca import CA
from cax.core.perceive.depthwise_conv_perceive import DepthwiseConvPerceive
from cax.core.update.elementary_update import ElementaryUpdate
from flax import nnx


@pytest.fixture
def elementary_update():
	"""Fixture to create an ElementaryUpdate instance."""
	return ElementaryUpdate()


def test_elementary_update_initialization():
	"""Test the initialization of ElementaryUpdate."""
	update = ElementaryUpdate()
	assert isinstance(update, ElementaryUpdate)
	assert update.patterns.shape == (8, 3)
	assert update.values.shape == (8,)


@pytest.mark.parametrize(
	"wolfram_code, expected_values",
	[
		("00000000", [0, 0, 0, 0, 0, 0, 0, 0]),
		("11111111", [1, 1, 1, 1, 1, 1, 1, 1]),
		("01101110", [0, 1, 1, 0, 1, 1, 1, 0]),  # Rule 110
	],
)
def test_elementary_update_wolfram_code(wolfram_code, expected_values):
	"""Test ElementaryUpdate with different Wolfram codes."""
	update = ElementaryUpdate(wolfram_code)
	assert jnp.array_equal(update.values, jnp.array(expected_values))


def test_elementary_update():
	"""Test the ElementaryUpdate in a CA simulation."""
	seed = 42
	spatial_dims = (16,)
	channel_size = 1
	wolfram_code = "01101110"  # Rule 110
	num_steps = 8

	rngs = nnx.Rngs(seed)

	state = jnp.zeros((*spatial_dims, channel_size))
	state = state.at[spatial_dims[0] // 2].set(1.0)

	perceive = DepthwiseConvPerceive(channel_size, rngs, num_kernels=3, kernel_size=(3,))
	update = ElementaryUpdate(wolfram_code)

	left_kernel = jnp.array([[1.0], [0.0], [0.0]])
	identity_kernel = jnp.array([[0.0], [1.0], [0.0]])
	right_kernel = jnp.array([[0.0], [0.0], [1.0]])

	kernel = jnp.concatenate([left_kernel, identity_kernel, right_kernel], axis=-1)
	kernel = jnp.expand_dims(kernel, axis=-2)
	perceive.depthwise_conv.kernel = nnx.Param(kernel)

	ca = CA(perceive, update)

	state = ca(state, num_steps=num_steps)

	expected_state = jnp.array(
		[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
	)
	assert jnp.array_equal(state, expected_state)
