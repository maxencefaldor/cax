"""Tests for the ResidualUpdate class."""

import jax.numpy as jnp
import pytest
from flax import nnx

from cax.core.ca import CA
from cax.core.perceive import ConvPerceive
from cax.core.update import ResidualUpdate


@pytest.fixture
def residual_update_no_input():
	"""Fixture to create a ResidualUpdate instance without input."""
	num_spatial_dims = 2
	channel_size = 3
	perception_size = 5
	hidden_layer_sizes = (32, 16)
	rngs = nnx.Rngs(0)
	return ResidualUpdate(
		num_spatial_dims,
		channel_size,
		perception_size,
		hidden_layer_sizes,
		rngs=rngs,
	)


@pytest.fixture
def residual_update_with_input():
	"""Fixture to create a ResidualUpdate instance with input."""
	num_spatial_dims = 2
	channel_size = 3
	perception_size = 5
	hidden_layer_sizes = (32, 16)
	input_size = 2
	rngs = nnx.Rngs(0)
	return ResidualUpdate(
		num_spatial_dims,
		channel_size,
		perception_size + input_size,
		hidden_layer_sizes,
		rngs=rngs,
	)


def test_residual_update_initialization(residual_update_no_input, residual_update_with_input):
	"""Test the initialization of ResidualUpdate."""
	assert isinstance(residual_update_no_input, ResidualUpdate)
	assert isinstance(residual_update_with_input, ResidualUpdate)
	assert len(residual_update_no_input.layers) == 3  # Input layer, hidden layer, output layer
	assert len(residual_update_with_input.layers) == 3
	assert residual_update_no_input.activation_fn == nnx.relu
	assert residual_update_with_input.activation_fn == nnx.relu


def test_residual_update_call_no_input(residual_update_no_input):
	"""Test the __call__ method of ResidualUpdate without input."""
	state = jnp.ones((8, 8, 3))
	perception = jnp.ones((8, 8, 5))
	updated_state = residual_update_no_input(state, perception, None)
	assert updated_state.shape == (8, 8, 3)
	assert jnp.all(jnp.isfinite(updated_state))


def test_residual_update_call_with_input(residual_update_with_input):
	"""Test the __call__ method of ResidualUpdate with input."""
	state = jnp.ones((8, 8, 3))
	perception = jnp.ones((8, 8, 5))
	input_data = jnp.ones((8, 8, 2))
	updated_state = residual_update_with_input(state, perception, input_data)
	assert updated_state.shape == (8, 8, 3)
	assert jnp.all(jnp.isfinite(updated_state))


def test_residual_update_in_ca():
	"""Test the ResidualUpdate in a CA simulation."""
	seed = 42
	spatial_dims = (16, 16)
	channel_size = 3
	num_kernels = 3
	hidden_layer_sizes = (32, 16)
	input_size = 2
	num_steps = 4

	rngs = nnx.Rngs(seed)

	state = jnp.zeros((*spatial_dims, channel_size))
	state = state.at[8:11, 8:11, :].set(1.0)

	perceive = ConvPerceive(
		channel_size=channel_size,
		perception_size=num_kernels * channel_size,
		rngs=rngs,
		feature_group_count=channel_size,
	)
	update = ResidualUpdate(
		num_spatial_dims=len(spatial_dims),
		channel_size=channel_size,
		perception_size=input_size + num_kernels * channel_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
	)

	ca = CA(perceive, update)

	input_data = jnp.ones((num_steps, *spatial_dims, input_size))
	final_state, metrics = ca(state, input_data, num_steps=num_steps, input_in_axis=0)

	assert final_state.shape == state.shape
	assert jnp.all(jnp.isfinite(final_state))


@pytest.mark.parametrize(
	"num_spatial_dims, channel_size, perception_size, hidden_layer_sizes, input_size",
	[
		(1, 2, 3, (16,), 1),
		(2, 4, 6, (32, 16), 2),
		(3, 5, 7, (64, 32, 16), 3),
	],
)
def test_residual_update_different_configs(
	num_spatial_dims, channel_size, perception_size, hidden_layer_sizes, input_size
):
	"""Test ResidualUpdate with different configurations."""
	rngs = nnx.Rngs(0)
	update = ResidualUpdate(
		num_spatial_dims=num_spatial_dims,
		channel_size=channel_size,
		perception_size=perception_size + input_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
	)
	assert isinstance(update, ResidualUpdate)

	spatial_dims = (8,) * num_spatial_dims
	state = jnp.ones((*spatial_dims, channel_size))
	perception = jnp.ones((*spatial_dims, perception_size))
	input_data = jnp.ones((*spatial_dims, input_size))
	updated_state = update(state, perception, input_data)

	assert updated_state.shape == state.shape
	assert jnp.all(jnp.isfinite(updated_state))
