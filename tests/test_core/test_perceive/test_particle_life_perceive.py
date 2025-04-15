"""Tests for the ParticleLifePerceive module."""

import jax.numpy as jnp
import pytest

from cax.core.perceive.particle_life_perceive import ParticleLifePerceive


@pytest.fixture
def particle_life_perceive():
	"""Fixture to create a ParticleLifePerceive instance."""
	A = jnp.array([[1.0, -0.5], [-0.5, 1.0]])  # Simple 2x2 attraction matrix
	return ParticleLifePerceive(A=A, r_max=0.15, beta=0.3)


def test_particle_life_perceive_initialization():
	"""Test the initialization of ParticleLifePerceive."""
	A = jnp.array([[1.0, -0.5], [-0.5, 1.0]])
	perceive = ParticleLifePerceive(A=A, r_max=0.2, beta=0.4, boundary="CIRCULAR")

	assert isinstance(perceive, ParticleLifePerceive)
	assert perceive.r_max == 0.2
	assert perceive.beta == 0.4
	assert perceive.boundary == "CIRCULAR"
	assert jnp.array_equal(perceive.A, A)


def test_get_forces(particle_life_perceive):
	"""Test the force calculation function."""
	distances = jnp.array([0.1, 0.2, 0.3])
	attraction_factors = jnp.array([1.0, -0.5, 0.0])

	forces = particle_life_perceive.get_forces(distances, attraction_factors)
	assert forces.shape == distances.shape

	# Test force at different distance regimes
	close_dist = jnp.array([0.01])  # < beta
	mid_dist = jnp.array([0.1])  # between beta and 1
	far_dist = jnp.array([0.2])  # > 1

	close_force = particle_life_perceive.get_forces(close_dist, jnp.array([1.0]))
	mid_force = particle_life_perceive.get_forces(mid_dist, jnp.array([1.0]))
	far_force = particle_life_perceive.get_forces(far_dist, jnp.array([1.0]))

	assert close_force[0] < 0  # Repulsive force
	assert mid_force[0] != 0  # Attractive/repulsive based on factor
	assert far_force[0] == 0  # No force beyond r_max


def test_particle_life_perceive_call():
	"""Test the __call__ method of ParticleLifePerceive."""
	A = jnp.array([[1.0, -0.5], [-0.5, 1.0]])
	perceive = ParticleLifePerceive(A=A)

	# Create a simple state with 2 particles
	class_ = jnp.array([0, 1])  # Two different particle types
	position = jnp.array([[0.2, 0.3], [0.7, 0.8]])  # 2D positions
	velocity = jnp.zeros((2, 2))  # Zero initial velocities

	state = (class_, position, velocity)
	direction_norm, forces = perceive(state)

	assert direction_norm.shape == (2, 2, 2)  # (num_particles, num_particles, dims)
	assert forces.shape == (2, 2)  # (num_particles, num_particles)


def test_periodic_boundary():
	"""Test periodic boundary conditions."""
	A = jnp.array([[1.0]])
	perceive = ParticleLifePerceive(A=A, boundary="CIRCULAR")

	# Create two particles near opposite boundaries
	class_ = jnp.array([0, 0])
	position = jnp.array([[0.1, 0.1], [0.9, 0.9]])  # Near opposite corners
	velocity = jnp.zeros((2, 2))

	state = (class_, position, velocity)
	direction_norm, forces = perceive(state)

	# Check that particles interact across boundary
	assert jnp.any(forces != 0)
