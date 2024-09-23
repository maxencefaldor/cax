"""Tests for the DepthwiseConvPerceive module."""

import jax
import jax.numpy as jnp
import pytest
from cax.core.perceive.depthwise_conv_perceive import DepthwiseConvPerceive
from flax import nnx


@pytest.fixture
def rngs():
	"""Fixture to provide random number generators."""
	return nnx.Rngs(0)


@pytest.fixture
def depthwise_conv_perceive(rngs):
	"""Fixture to provide a DepthwiseConvPerceive instance."""
	return DepthwiseConvPerceive(channel_size=4, rngs=rngs)


def test_depthwise_conv_perceive_initialization(depthwise_conv_perceive):
	"""Test the initialization of DepthwiseConvPerceive."""
	assert isinstance(depthwise_conv_perceive, DepthwiseConvPerceive)
	assert isinstance(depthwise_conv_perceive.depthwise_conv, nnx.Conv)


def test_depthwise_conv_perceive_output_shape(depthwise_conv_perceive):
	"""Test the output shape of DepthwiseConvPerceive."""
	state = jnp.zeros((10, 10, 4))
	perception = depthwise_conv_perceive(state)
	assert perception.shape == (10, 10, 12)  # 4 channels * 3 kernels = 12


def test_depthwise_conv_perceive_custom_params(rngs):
	"""Test DepthwiseConvPerceive with custom parameters."""
	custom_perceive = DepthwiseConvPerceive(channel_size=8, rngs=rngs, num_kernels=5, kernel_size=(5, 5), use_bias=True)
	assert custom_perceive.depthwise_conv.out_features == 40  # 8 channels * 5 kernels
	assert custom_perceive.depthwise_conv.kernel_size == (5, 5)
	assert custom_perceive.depthwise_conv.use_bias


def test_depthwise_conv_perceive_forward_pass(depthwise_conv_perceive):
	"""Test the forward pass of DepthwiseConvPerceive."""
	key = jax.random.key(0)
	state = jax.random.normal(key, (10, 10, 4))
	perception = depthwise_conv_perceive(state)
	assert jnp.any(perception != 0)  # Ensure non-zero output


@pytest.mark.parametrize("channel_size,num_kernels", [(2, 2), (4, 3), (8, 5)])
def test_depthwise_conv_perceive_different_sizes(rngs, channel_size, num_kernels):
	"""Test DepthwiseConvPerceive with different sizes."""
	perceive = DepthwiseConvPerceive(channel_size=channel_size, rngs=rngs, num_kernels=num_kernels)
	state = jnp.zeros((5, 5, channel_size))
	perception = perceive(state)
	assert perception.shape == (5, 5, channel_size * num_kernels)
