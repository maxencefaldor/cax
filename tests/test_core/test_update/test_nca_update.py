"""Tests for the NCAUpdate class."""

import jax.numpy as jnp
import pytest
from cax.core.ca import CA
from cax.core.perceive.conv_perceive import ConvPerceive
from cax.core.update.nca_update import NCAUpdate
from flax import nnx


@pytest.fixture
def nca_update_no_input():
	"""Fixture to create an NCAUpdate instance without input."""
	channel_size = 3
	perception_size = 5
	hidden_layer_sizes = (32, 16)
	rngs = nnx.Rngs(0)
	return NCAUpdate(
		channel_size=channel_size,
		perception_size=perception_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
	)


@pytest.fixture
def nca_update_with_input():
	"""Fixture to create an NCAUpdate instance with input."""
	channel_size = 3
	perception_size = 5
	hidden_layer_sizes = (32, 16)
	input_size = 2
	rngs = nnx.Rngs(0)
	return NCAUpdate(
		channel_size=channel_size,
		perception_size=perception_size + input_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
	)


def test_nca_update_initialization(nca_update_no_input, nca_update_with_input):
	"""Test the initialization of NCAUpdate."""
	assert isinstance(nca_update_no_input, NCAUpdate)
	assert isinstance(nca_update_with_input, NCAUpdate)
	assert len(nca_update_no_input.layers) == 3  # Input layer, hidden layer, output layer
	assert len(nca_update_with_input.layers) == 3
	assert nca_update_no_input.activation_fn == nnx.relu
	assert nca_update_with_input.activation_fn == nnx.relu
	assert nca_update_no_input.alive_threshold == 0.1
	assert nca_update_with_input.alive_threshold == 0.1


def test_nca_update_call_no_input(nca_update_no_input):
	"""Test the __call__ method of NCAUpdate without input."""
	state = jnp.ones((8, 8, 3))
	perception = jnp.ones((8, 8, 5))
	updated_state = nca_update_no_input(state, perception, None)
	assert updated_state.shape == (8, 8, 3)
	assert jnp.all(jnp.isfinite(updated_state))


def test_nca_update_call_with_input(nca_update_with_input):
	"""Test the __call__ method of NCAUpdate with input."""
	state = jnp.ones((8, 8, 3))
	perception = jnp.ones((8, 8, 5))
	input_data = jnp.ones((8, 8, 2))
	updated_state = nca_update_with_input(state, perception, input_data)
	assert updated_state.shape == (8, 8, 3)
	assert jnp.all(jnp.isfinite(updated_state))


def test_nca_update_in_ca():
	"""Test the NCAUpdate in a CA simulation."""
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
	update = NCAUpdate(
		channel_size=channel_size,
		perception_size=input_size + num_kernels * channel_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
	)

	ca = CA(perceive, update)

	input_data = jnp.ones((num_steps, *spatial_dims, input_size))
	final_state = ca(state, input_data, num_steps=num_steps, input_in_axis=0)

	assert final_state.shape == state.shape
	assert jnp.all(jnp.isfinite(final_state))
	assert jnp.all((final_state >= 0) & (final_state <= 1))


@pytest.mark.parametrize(
	"channel_size, perception_size, hidden_layer_sizes, alive_threshold",
	[
		(2, 4, (16,), 0.05),
		(4, 6, (32, 16), 0.1),
		(5, 7, (64, 32, 16), 0.2),
	],
)
def test_nca_update_different_configs(channel_size, perception_size, hidden_layer_sizes, alive_threshold):
	"""Test NCAUpdate with different configurations."""
	rngs = nnx.Rngs(0)
	update = NCAUpdate(
		channel_size=channel_size,
		perception_size=perception_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
		alive_threshold=alive_threshold,
	)
	assert isinstance(update, NCAUpdate)

	spatial_dims = (8, 8)
	state = jnp.ones((*spatial_dims, channel_size))
	perception = jnp.ones((*spatial_dims, perception_size))
	updated_state = update(state, perception, None)

	assert updated_state.shape == (*spatial_dims, channel_size)
	assert jnp.all(jnp.isfinite(updated_state))
	assert jnp.all((updated_state >= 0) & (updated_state <= 1))


def test_nca_update_get_alive_mask():
	"""Test the get_alive_mask method of NCAUpdate."""
	channel_size = 3
	perception_size = 5
	hidden_layer_sizes = (32, 16)
	rngs = nnx.Rngs(0)
	update = NCAUpdate(
		channel_size=channel_size,
		perception_size=perception_size,
		hidden_layer_sizes=hidden_layer_sizes,
		rngs=rngs,
		alive_threshold=0.1,
	)

	state = jnp.array([[[0.05, 0.05, 0.05], [0.15, 0.15, 0.15]], [[0.25, 0.25, 0.25], [0.35, 0.35, 0.35]]])

	alive_mask = update.get_alive_mask(state)
	expected_mask = jnp.array([[[True], [True]], [[True], [True]]])

	assert jnp.array_equal(alive_mask, expected_mask)
