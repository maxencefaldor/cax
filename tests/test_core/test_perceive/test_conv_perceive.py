"""Tests for the ConvPerceive module."""

import jax
import jax.numpy as jnp
import pytest
from cax.core.perceive.conv_perceive import ConvPerceive
from flax import nnx


@pytest.fixture
def rngs():
	"""Fixture to provide random number generators."""
	return nnx.Rngs(0)


@pytest.fixture
def conv_perceive(rngs):
	"""Fixture to provide a ConvPerceive instance."""
	return ConvPerceive(channel_size=4, perception_size=8, hidden_layer_sizes=(16,), rngs=rngs)


def test_conv_perceive_initialization(conv_perceive):
	"""Test the initialization of ConvPerceive."""
	assert isinstance(conv_perceive, ConvPerceive)
	assert len(conv_perceive.layers) == 2
	assert all(isinstance(layer, nnx.Conv) for layer in conv_perceive.layers)


def test_conv_perceive_output_shape(conv_perceive):
	"""Test the output shape of ConvPerceive."""
	key = jax.random.key(0)
	state = jax.random.normal(key, (10, 10, 4))
	perception = conv_perceive(state)
	assert perception.shape == (10, 10, 8)


def test_conv_perceive_custom_params(rngs):
	"""Test ConvPerceive with custom parameters."""
	custom_perceive = ConvPerceive(
		channel_size=8,
		perception_size=16,
		hidden_layer_sizes=(32, 64),
		rngs=rngs,
		kernel_size=(5, 5),
		use_bias=True,
		activation_fn=jax.nn.tanh,
	)
	assert len(custom_perceive.layers) == 3
	assert custom_perceive.layers[0].kernel_size == (5, 5)
	assert custom_perceive.layers[0].use_bias
	assert custom_perceive.activation_fn == jax.nn.tanh


def test_conv_perceive_forward_pass(conv_perceive):
	"""Test the forward pass of ConvPerceive."""
	key = jax.random.key(0)
	state = jax.random.normal(key, (10, 10, 4))
	perception = conv_perceive(state)
	assert jnp.any(perception != 0)  # Ensure non-zero output


@pytest.mark.parametrize(
	"channel_size,perception_size,hidden_sizes", [(2, 4, (8,)), (4, 8, (16, 32)), (8, 16, (32, 64, 128))]
)
def test_conv_perceive_different_sizes(rngs, channel_size, perception_size, hidden_sizes):
	"""Test ConvPerceive with different sizes."""
	perceive = ConvPerceive(
		channel_size=channel_size, perception_size=perception_size, hidden_layer_sizes=hidden_sizes, rngs=rngs
	)
	state = jnp.zeros((5, 5, channel_size))
	perception = perceive(state)
	assert perception.shape == (5, 5, perception_size)
