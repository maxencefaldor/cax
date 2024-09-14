"""Tests for the DWConvPerceive module."""

import jax
import jax.numpy as jnp
import pytest
from cax.core.perceive.dwconv_perceive import DWConvPerceive
from flax import nnx


@pytest.fixture
def rngs():
	"""Fixture to provide random number generators."""
	return nnx.Rngs(0)


@pytest.fixture
def dwconv_perceive(rngs):
	"""Fixture to provide a DWConvPerceive instance."""
	return DWConvPerceive(channel_size=4, rngs=rngs)


def test_dwconv_perceive_initialization(dwconv_perceive):
	"""Test the initialization of DWConvPerceive."""
	assert isinstance(dwconv_perceive, DWConvPerceive)
	assert isinstance(dwconv_perceive.dwconv, nnx.Conv)


def test_dwconv_perceive_output_shape(dwconv_perceive):
	"""Test the output shape of DWConvPerceive."""
	state = jnp.zeros((10, 10, 4))
	perception = dwconv_perceive(state)
	assert perception.shape == (10, 10, 12)  # 4 channels * 3 kernels = 12


def test_dwconv_perceive_custom_params(rngs):
	"""Test DWConvPerceive with custom parameters."""
	custom_perceive = DWConvPerceive(channel_size=8, rngs=rngs, num_kernels=5, kernel_size=(5, 5), use_bias=True)
	assert custom_perceive.dwconv.out_features == 40  # 8 channels * 5 kernels
	assert custom_perceive.dwconv.kernel_size == (5, 5)
	assert custom_perceive.dwconv.use_bias


def test_dwconv_perceive_forward_pass(dwconv_perceive):
	"""Test the forward pass of DWConvPerceive."""
	key = jax.random.PRNGKey(0)
	state = jax.random.normal(key, (10, 10, 4))
	perception = dwconv_perceive(state)
	assert jnp.any(perception != 0)  # Ensure non-zero output


@pytest.mark.parametrize("channel_size,num_kernels", [(2, 2), (4, 3), (8, 5)])
def test_dwconv_perceive_different_sizes(rngs, channel_size, num_kernels):
	"""Test DWConvPerceive with different sizes."""
	perceive = DWConvPerceive(channel_size=channel_size, rngs=rngs, num_kernels=num_kernels)
	state = jnp.zeros((5, 5, channel_size))
	perception = perceive(state)
	assert perception.shape == (5, 5, channel_size * num_kernels)
