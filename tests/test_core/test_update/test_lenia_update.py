"""Tests for the LeniaUpdate class."""

import jax.numpy as jnp
import pytest
from cax.core.ca import CA
from cax.core.perceive.lenia_perceive import LeniaPerceive
from cax.core.update.lenia_update import LeniaUpdate, growth


@pytest.fixture
def lenia_config():
	"""Fixture to provide a sample Lenia configuration."""
	return {
		"state_size": 64,
		"channel_size": 3,
		"R": 12,
		"T": 2,
		"state_scale": 1,
		"kernel_params": [
			{"b": [1], "m": 0.3, "s": 0.1, "h": 0.1, "r": 1.0, "c0": 0, "c1": 0},
			{"b": [1], "m": 0.5, "s": 0.2, "h": 0.2, "r": 0.8, "c0": 1, "c1": 1},
			{"b": [1], "m": 0.7, "s": 0.3, "h": 0.3, "r": 0.6, "c0": 2, "c1": 2},
		],
	}


@pytest.fixture
def lenia_update(lenia_config):
	"""Fixture to create a LeniaUpdate instance."""
	return LeniaUpdate(lenia_config)


def test_lenia_update_initialization(lenia_update, lenia_config):
	"""Test the initialization of LeniaUpdate."""
	assert isinstance(lenia_update, LeniaUpdate)
	assert lenia_update._config == lenia_config
	assert lenia_update.m.shape == (1, 1, 3)
	assert lenia_update.s.shape == (1, 1, 3)
	assert lenia_update.h.shape == (1, 1, 3)
	assert lenia_update.reshape_k_c.shape == (3, 3)


def test_growth_function():
	"""Test the growth function."""
	x = jnp.array([0.0, 0.5, 1.0])
	mean = 0.5
	stdev = 0.1
	result = growth(x, mean, stdev)
	expected = jnp.array([-0.99999255, 1.0, -0.99999255])
	assert jnp.allclose(result, expected, atol=1e-5)


def test_lenia_update_call(lenia_update):
	"""Test the __call__ method of LeniaUpdate."""
	state = jnp.ones((64, 64, 3)) * 0.5
	perception = jnp.ones((64, 64, 3)) * 0.5
	updated_state = lenia_update(state, perception, None)
	assert updated_state.shape == (64, 64, 3)
	assert jnp.all(jnp.isfinite(updated_state))
	assert jnp.all((updated_state >= 0) & (updated_state <= 1))


def test_lenia_update_in_ca(lenia_config):
	"""Test the LeniaUpdate in a CA simulation."""
	num_steps = 10

	state = jnp.zeros((lenia_config["state_size"], lenia_config["state_size"], lenia_config["channel_size"]))
	state = state.at[32:34, 32:34].set(1.0)

	perceive = LeniaPerceive(lenia_config)
	update = LeniaUpdate(lenia_config)

	ca = CA(perceive, update)

	final_state = ca(state, num_steps=num_steps)

	assert final_state.shape == state.shape
	assert jnp.all(jnp.isfinite(final_state))
	assert jnp.all((final_state >= 0) & (final_state <= 1))
	assert not jnp.array_equal(final_state, state)


@pytest.mark.parametrize(
	"config_update",
	[
		{"R": 8, "T": 1},
		{"state_size": 32, "channel_size": 3},
		{
			"kernel_params": [
				{"b": [1], "m": 0.4, "s": 0.15, "h": 0.15, "r": 0.9, "c0": 0, "c1": 0},
				{"b": [1], "m": 0.5, "s": 0.2, "h": 0.2, "r": 0.8, "c0": 1, "c1": 1},
				{"b": [1], "m": 0.6, "s": 0.25, "h": 0.25, "r": 0.7, "c0": 2, "c1": 2},
			]
		},
	],
)
def test_lenia_update_different_configs(lenia_config, config_update):
	"""Test LeniaUpdate with different configurations."""
	updated_config = {**lenia_config, **config_update}
	update = LeniaUpdate(updated_config)
	assert isinstance(update, LeniaUpdate)

	state_shape = (updated_config["state_size"], updated_config["state_size"], updated_config["channel_size"])
	state = jnp.ones(state_shape) * 0.5
	perception = jnp.ones(state_shape) * 0.5
	updated_state = update(state, perception, None)

	assert updated_state.shape == state_shape
	assert jnp.all(jnp.isfinite(updated_state))
	assert jnp.all((updated_state >= 0) & (updated_state <= 1))
