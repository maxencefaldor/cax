"""Tests for the VonNeumannPerceive module."""

import jax.numpy as jnp
import pytest
from cax.core.perceive.von_neumann_perceive import VonNeumannPerceive


@pytest.fixture
def von_neumann_perceive_2d():
	"""Fixture to create a 2D VonNeumannPerceive instance."""
	return VonNeumannPerceive(num_spatial_dims=2, radius=1)


@pytest.fixture
def von_neumann_perceive_3d():
	"""Fixture to create a 3D VonNeumannPerceive instance."""
	return VonNeumannPerceive(num_spatial_dims=3, radius=1)


def test_von_neumann_perceive_initialization():
	"""Test the initialization of VonNeumannPerceive."""
	perceive = VonNeumannPerceive(num_spatial_dims=2, radius=1)
	assert isinstance(perceive, VonNeumannPerceive)
	assert perceive.num_spatial_dims == 2
	assert perceive.radius == 1


def test_von_neumann_perceive_2d(von_neumann_perceive_2d):
	"""Test VonNeumannPerceive for 2D input."""
	state = jnp.arange(16).reshape(4, 4, 1)
	perception = von_neumann_perceive_2d(state)

	assert perception.shape == (4, 4, 5)

	# Check the perception for the center cell (1, 1)
	expected_center = jnp.array([5, 9, 6, 4, 1])
	assert jnp.allclose(jnp.sum(perception[1, 1]), jnp.sum(expected_center))


def test_von_neumann_perceive_3d(von_neumann_perceive_3d):
	"""Test VonNeumannPerceive for 3D input."""
	state = jnp.arange(27).reshape(3, 3, 3, 1)
	perception = von_neumann_perceive_3d(state)

	assert perception.shape == (3, 3, 3, 7)

	# Check the perception for the center cell (1, 1, 1)
	expected_center = jnp.array([13, 12, 14, 10, 16, 4, 22])
	assert jnp.allclose(jnp.sum(perception[1, 1, 1]), jnp.sum(expected_center))


def test_von_neumann_perceive_different_radius():
	"""Test VonNeumannPerceive with different radius values."""
	perceive = VonNeumannPerceive(num_spatial_dims=2, radius=2)
	state = jnp.arange(25).reshape(5, 5, 1)
	perception = perceive(state)

	assert perception.shape == (5, 5, 13)

	# Check the perception for the center cell (2, 2)
	expected_center = jnp.array([12, 22, 18, 17, 16, 14, 13, 11, 10, 8, 7, 6, 2])
	assert jnp.allclose(jnp.sum(perception[2, 2]), jnp.sum(expected_center))


def test_von_neumann_perceive_1d():
	"""Test VonNeumannPerceive for 1D input."""
	perceive = VonNeumannPerceive(num_spatial_dims=1, radius=1)
	state = jnp.arange(5).reshape(5, 1)
	perception = perceive(state)

	assert perception.shape == (5, 3)

	# Check the perception for the center cell (2)
	expected_center = jnp.array([2, 3, 1])
	assert jnp.allclose(jnp.sum(perception[2]), jnp.sum(expected_center))


def test_von_neumann_perceive_multi_channel():
	"""Test VonNeumannPerceive with multi-channel input."""
	perceive = VonNeumannPerceive(num_spatial_dims=2, radius=1)
	state = jnp.arange(32).reshape(4, 4, 2)
	perception = perceive(state)

	assert perception.shape == (4, 4, 10)

	# Check the perception for the center cell (1, 1)
	expected_center = jnp.array([10, 11, 18, 19, 12, 13, 8, 9, 2, 3])
	assert jnp.allclose(jnp.sum(perception[1, 1]), jnp.sum(expected_center))


def test_von_neumann_perceive_edge_cases():
	"""Test VonNeumannPerceive for edge cases."""
	perceive = VonNeumannPerceive(num_spatial_dims=2, radius=1)
	state = jnp.arange(16).reshape(4, 4, 1)
	perception = perceive(state)

	# Check the perception for the top-left corner (0, 0)
	expected_corner = jnp.array([0, 4, 1, 3, 12])
	assert jnp.allclose(jnp.sum(perception[0, 0]), jnp.sum(expected_corner))

	# Check the perception for the bottom-right corner (3, 3)
	expected_corner = jnp.array([15, 3, 12, 14, 11])
	assert jnp.allclose(jnp.sum(perception[3, 3]), jnp.sum(expected_corner))
