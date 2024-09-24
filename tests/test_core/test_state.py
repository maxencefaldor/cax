"""Tests for the state module."""

import jax.numpy as jnp
import pytest
from cax.core.state import state_from_rgba_to_rgb, state_to_alive, state_to_rgba
from jax import Array


@pytest.fixture
def init_state() -> Array:
	"""Initialize a test state."""
	return jnp.array([[[0.1, 0.2, 0.3, 0.4, 0.5]], [[0.6, 0.7, 0.8, 0.9, 1.0]]])


def test_state_to_alive(init_state: Array) -> None:
	"""Test the state_to_alive function."""
	alive = state_to_alive(init_state)
	expected = jnp.array([[[0.5]], [[1.0]]])
	assert jnp.allclose(alive, expected)


def test_state_to_rgba(init_state: Array) -> None:
	"""Test the state_to_rgba function."""
	rgba = state_to_rgba(init_state)
	expected = jnp.array([[[0.2, 0.3, 0.4, 0.5]], [[0.7, 0.8, 0.9, 1.0]]])
	assert jnp.allclose(rgba, expected)


def test_state_from_rgba_to_rgb(init_state: Array) -> None:
	"""Test the state_from_rgba_to_rgb function."""
	rgb = state_from_rgba_to_rgb(init_state)
	expected = jnp.array([[[0.7, 0.8, 0.9]], [[0.7, 0.8, 0.9]]])
	assert jnp.allclose(rgb, expected)


def test_state_from_rgba_to_rgb_clipping(init_state: Array) -> None:
	"""Test the state_from_rgba_to_rgb function with clipping."""
	state = jnp.array([[[0.1, 0.2, 0.3, 0.4, 1.5]], [[0.6, 0.7, 0.8, 0.9, -0.5]]])
	rgb = state_from_rgba_to_rgb(state)
	expected = jnp.array([[[0.2, 0.3, 0.4]], [[1.7, 1.8, 1.9]]])
	assert jnp.allclose(rgb, expected)
