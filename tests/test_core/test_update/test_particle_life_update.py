"""Tests for the ParticleLifeUpdate class."""

import jax
import jax.numpy as jnp
import pytest

from cax.core.ca import CA
from cax.core.perceive.particle_life_perceive import ParticleLifePerceive
from cax.core.update.particle_life_update import ParticleLifeUpdate


@pytest.fixture
def particle_life_update():
	"""Fixture to create a ParticleLifeUpdate instance."""
	return ParticleLifeUpdate(dt=0.1, velocity_half_life=0.1, boundary="CIRCULAR")


def test_particle_life_update_initialization():
	"""Test the initialization of ParticleLifeUpdate."""
	update = ParticleLifeUpdate(dt=0.05, velocity_half_life=0.2, boundary="CIRCULAR")
	assert isinstance(update, ParticleLifeUpdate)
	assert update.friction_factor == float(jnp.power(0.5, 0.05 / 0.2))
	assert update.dt == 0.05
	assert update.boundary == "CIRCULAR"


def test_particle_life_update_call():
	"""Test the __call__ method of ParticleLifeUpdate."""
	update = ParticleLifeUpdate()

	# Create a simple state with 2 particles
	class_ = jnp.array([0, 1])
	position = jnp.array([[0.2, 0.3], [0.7, 0.8]])
	velocity = jnp.array([[0.1, 0.1], [-0.1, -0.1]])
	state = (class_, position, velocity)

	# Create perception data
	direction_norm = jnp.array([[[0.1, 0.1], [-0.1, -0.1]], [[0.1, 0.1], [-0.1, -0.1]]])
	forces = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
	perception = (direction_norm, forces)

	new_state = update(state, perception, None)
	new_class, new_position, new_velocity = new_state

	assert new_class.shape == class_.shape
	assert new_position.shape == position.shape
	assert new_velocity.shape == velocity.shape
	assert jnp.all(jnp.isfinite(new_position))
	assert jnp.all(jnp.isfinite(new_velocity))


def test_particle_life_update_in_ca():
	"""Test the ParticleLifeUpdate in a CA simulation."""
	key = jax.random.PRNGKey(42)
	num_particles = 10
	num_steps = 4

	# Initialize state
	class_ = jnp.zeros(num_particles, dtype=jnp.int32)
	position = jnp.array(jax.random.uniform(key, shape=(num_particles, 2)))
	velocity = jnp.zeros((num_particles, 2))
	state = (class_, position, velocity)

	# Create CA components
	A = jnp.array([[1.0]])  # Simple attraction matrix
	perceive = ParticleLifePerceive(A=A)
	update = ParticleLifeUpdate()

	ca = CA(perceive, update)
	final_state = ca(state, num_steps=num_steps)

	# Check output shapes and values
	final_class, final_position, final_velocity = final_state
	assert final_class.shape == class_.shape
	assert final_position.shape == position.shape
	assert final_velocity.shape == velocity.shape
	assert jnp.all(jnp.isfinite(final_position))
	assert jnp.all(jnp.isfinite(final_velocity))
	assert jnp.all((final_position >= 0.0) & (final_position <= 1.0))


def test_boundary_conditions():
	"""Test boundary conditions in ParticleLifeUpdate."""
	update = ParticleLifeUpdate(boundary="CIRCULAR")

	# Create a particle near the boundary
	class_ = jnp.array([0])
	position = jnp.array([[0.95, 0.95]])
	velocity = jnp.array([[0.1, 0.1]])
	state = (class_, position, velocity)

	# Create minimal perception data
	direction_norm = jnp.array([[[0.0, 0.0]]])
	forces = jnp.array([[0.0]])
	perception = (direction_norm, forces)

	new_state = update(state, perception, None)
	_, new_position, _ = new_state

	# Check that position wraps around
	assert jnp.all((new_position >= 0.0) & (new_position <= 1.0))


def test_friction():
	"""Test friction in ParticleLifeUpdate."""
	velocity_half_life = 0.5
	update = ParticleLifeUpdate(velocity_half_life=velocity_half_life)

	# Create a particle with initial velocity
	class_ = jnp.array([0])
	position = jnp.array([[0.5, 0.5]])
	velocity = jnp.array([[1.0, 1.0]])
	state = (class_, position, velocity)

	# Create minimal perception data
	direction_norm = jnp.array([[[0.0, 0.0]]])
	forces = jnp.array([[0.0]])
	perception = (direction_norm, forces)

	new_state = update(state, perception, None)
	_, _, new_velocity = new_state

	# Check that velocity magnitude has decreased
	assert jnp.all(jnp.abs(new_velocity) < jnp.abs(velocity))
