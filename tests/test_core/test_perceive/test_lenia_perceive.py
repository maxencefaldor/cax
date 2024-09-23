"""Tests for the LeniaPerceive class and related functions."""

import jax
import jax.numpy as jnp
import pytest
from cax.core.perceive.lenia_perceive import LeniaPerceive, bell


@pytest.fixture
def lenia_config():
	"""Fixture to provide a sample Lenia configuration."""
	return {
		"state_size": 128,
		"channel_size": 3,
		"R": 12,
		"T": 2,
		"state_scale": 1,
		"kernel_params": [
			{"b": [1], "m": 0.272, "s": 0.0595, "h": 0.138, "r": 0.91, "c0": 0, "c1": 0},
			{"b": [1], "m": 0.349, "s": 0.1585, "h": 0.48, "r": 0.62, "c0": 0, "c1": 0},
			{"b": [1, 1 / 4], "m": 0.2, "s": 0.0332, "h": 0.284, "r": 0.5, "c0": 0, "c1": 0},
			{"b": [0, 1], "m": 0.114, "s": 0.0528, "h": 0.256, "r": 0.97, "c0": 1, "c1": 1},
			{"b": [1], "m": 0.447, "s": 0.0777, "h": 0.5, "r": 0.72, "c0": 1, "c1": 1},
			{"b": [5 / 6, 1], "m": 0.247, "s": 0.0342, "h": 0.622, "r": 0.8, "c0": 1, "c1": 1},
			{"b": [1], "m": 0.21, "s": 0.0617, "h": 0.35, "r": 0.96, "c0": 2, "c1": 2},
			{"b": [1], "m": 0.462, "s": 0.1192, "h": 0.218, "r": 0.56, "c0": 2, "c1": 2},
			{"b": [1], "m": 0.446, "s": 0.1793, "h": 0.556, "r": 0.78, "c0": 2, "c1": 2},
			{"b": [11 / 12, 1], "m": 0.327, "s": 0.1408, "h": 0.344, "r": 0.79, "c0": 0, "c1": 1},
			{"b": [3 / 4, 1], "m": 0.476, "s": 0.0995, "h": 0.456, "r": 0.5, "c0": 0, "c1": 2},
			{"b": [11 / 12, 1], "m": 0.379, "s": 0.0697, "h": 0.67, "r": 0.72, "c0": 1, "c1": 0},
			{"b": [1], "m": 0.262, "s": 0.0877, "h": 0.42, "r": 0.68, "c0": 1, "c1": 2},
			{"b": [1 / 6, 1, 0], "m": 0.412, "s": 0.1101, "h": 0.43, "r": 0.82, "c0": 2, "c1": 0},
			{"b": [1], "m": 0.201, "s": 0.0786, "h": 0.278, "r": 0.82, "c0": 2, "c1": 1},
		],
	}


@pytest.fixture
def lenia_perceive(lenia_config):
	"""Fixture to provide a LeniaPerceive instance."""
	return LeniaPerceive(lenia_config)


def test_bell_function():
	"""Test the bell function."""
	x = jnp.array([0.0, 0.5, 1.0])
	mean = 0.0
	stdev = 1.0
	result = bell(x, mean, stdev)
	expected = jnp.array([1.0, 0.8824969, 0.60653067])
	assert jnp.allclose(result, expected, atol=1e-5)


def test_lenia_perceive_initialization(lenia_perceive, lenia_config):
	"""Test the initialization of LeniaPerceive."""
	assert isinstance(lenia_perceive, LeniaPerceive)
	assert lenia_perceive._config == lenia_config
	assert lenia_perceive.kernel_fft.shape == (128, 128, 15)  # Updated shape
	assert lenia_perceive.reshape_c_k.shape == (3, 15)  # Updated shape


def test_lenia_perceive_call(lenia_perceive):
	"""Test the __call__ method of LeniaPerceive."""
	key = jax.random.key(0)
	state = jax.random.uniform(key, (128, 128, 3))  # Updated shape
	perception = lenia_perceive(state)
	assert perception.shape == (128, 128, 15)  # Updated shape
	assert jnp.all(jnp.isfinite(perception))


def test_lenia_perceive_different_config():
	"""Test LeniaPerceive with a different configuration."""
	config = {
		"kernel_params": [
			{"b": [1], "m": 0.5, "s": 0.1, "r": 1.0, "c0": 0, "c1": 0},
			{"b": [1, 0.5], "m": 0.3, "s": 0.2, "r": 0.8, "c0": 1, "c1": 1},
			{"b": [0.5, 1], "m": 0.4, "s": 0.15, "r": 0.9, "c0": 2, "c1": 2},
		],
		"R": 15,
		"state_scale": 2,
		"state_size": 64,
		"channel_size": 3,
	}
	perceive = LeniaPerceive(config)
	assert perceive.kernel_fft.shape == (64, 64, 3)
	assert perceive.reshape_c_k.shape == (3, 3)

	key = jax.random.key(0)
	state = jax.random.uniform(key, (64, 64, 3))
	perception = perceive(state)
	assert perception.shape == (64, 64, 3)
	assert jnp.all(jnp.isfinite(perception))


@pytest.mark.parametrize("state_size,channel_size", [(16, 1), (32, 2), (64, 4)])
def test_lenia_perceive_different_sizes(state_size, channel_size):
	"""Test LeniaPerceive with different state and channel sizes."""
	config = {
		"kernel_params": [{"b": [1], "m": 0.5, "s": 0.1, "r": 1.0, "c0": i, "c1": i} for i in range(channel_size)],
		"R": 10,
		"state_scale": 1,
		"state_size": state_size,
		"channel_size": channel_size,
	}
	perceive = LeniaPerceive(config)
	key = jax.random.key(0)
	state = jax.random.uniform(key, (state_size, state_size, channel_size))
	perception = perceive(state)
	assert perception.shape == (state_size, state_size, channel_size)
	assert jnp.all(jnp.isfinite(perception))
