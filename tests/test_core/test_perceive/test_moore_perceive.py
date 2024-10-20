"""Tests for the MoorePerceive module."""

import jax.numpy as jnp
import pytest
from cax.core.perceive.moore_perceive import MoorePerceive


@pytest.fixture
def moore_perceive_2d():
	"""Fixture to create a 2D MoorePerceive instance."""
	return MoorePerceive(num_spatial_dims=2, radius=1)


@pytest.fixture
def moore_perceive_3d():
	"""Fixture to create a 3D MoorePerceive instance."""
	return MoorePerceive(num_spatial_dims=3, radius=1)


def test_moore_perceive_initialization():
	"""Test the initialization of MoorePerceive."""
	perceive = MoorePerceive(num_spatial_dims=2, radius=1)
	assert isinstance(perceive, MoorePerceive)
	assert perceive.num_spatial_dims == 2
	assert perceive.radius == 1


def test_moore_perceive_2d(moore_perceive_2d):
	"""Test MoorePerceive for 2D input."""
	state = jnp.arange(16).reshape(4, 4, 1)
	perception = moore_perceive_2d(state)

	assert perception.shape == (4, 4, 9)

	# Check the perception for the center cell (1, 1)
	expected_center = jnp.array([5, 10, 9, 8, 6, 4, 2, 1, 0])
	assert jnp.allclose(jnp.sum(perception[1, 1]), jnp.sum(expected_center))


def test_moore_perceive_3d(moore_perceive_3d):
	"""Test MoorePerceive for 3D input."""
	state = jnp.arange(27).reshape(3, 3, 3, 1)
	perception = moore_perceive_3d(state)

	assert perception.shape == (3, 3, 3, 27)


def test_moore_perceive_different_radius():
	"""Test MoorePerceive with different radius values."""
	perceive = MoorePerceive(num_spatial_dims=2, radius=2)
	state = jnp.arange(25).reshape(5, 5, 1)
	perception = perceive(state)

	assert perception.shape == (5, 5, 25)


def test_moore_perceive_1d():
	"""Test MoorePerceive for 1D input."""
	perceive = MoorePerceive(num_spatial_dims=1, radius=1)
	state = jnp.arange(5).reshape(5, 1)
	perception = perceive(state)

	assert perception.shape == (5, 3)

	# Check the perception for the center cell (2)
	expected_center = jnp.array([2, 3, 1])
	assert jnp.allclose(jnp.sum(perception[2]), jnp.sum(expected_center))


def test_moore_perceive_multi_channel():
	"""Test MoorePerceive with multi-channel input."""
	perceive = MoorePerceive(num_spatial_dims=2, radius=1)
	state = jnp.arange(32).reshape(4, 4, 2)
	perception = perceive(state)

	assert perception.shape == (4, 4, 18)

	# Check the perception for the center cell (1, 1)
	expected_center = jnp.array([10, 11, 20, 21, 18, 19, 16, 17, 12, 13, 8, 9, 4, 5, 2, 3, 0, 1])
	assert jnp.allclose(jnp.sum(perception[1, 1]), jnp.sum(expected_center))


def test_moore_perceive_edge_cases():
	"""Test MoorePerceive for edge cases."""
	perceive = MoorePerceive(num_spatial_dims=2, radius=1)
	state = jnp.arange(16).reshape(4, 4, 1)
	perception = perceive(state)

	# Check the perception for the top-left corner (0, 0)
	expected_corner = jnp.array([0, 5, 4, 7, 1, 3, 13, 12, 15])
	assert jnp.allclose(jnp.sum(perception[0, 0]), jnp.sum(expected_corner))

	# Check the perception for the bottom-right corner (3, 3)
	expected_corner = jnp.array([15, 0, 3, 2, 12, 14, 8, 11, 10])
	assert jnp.allclose(jnp.sum(perception[3, 3]), jnp.sum(expected_corner))
