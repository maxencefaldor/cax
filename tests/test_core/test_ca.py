"""Tests for the CA class."""

import jax.numpy as jnp
import pytest
from flax import nnx

from cax.core import CA
from cax.core.perceive import Perceive
from cax.core.update import Update


class DummyPerceive(Perceive):
	"""A dummy perceive class for testing."""

	def __call__(self, state):
		"""Multiply the state by 2."""
		return state * 2


class DummyUpdate(Update):
	"""A dummy update class for testing."""

	def __call__(self, state, perception, input=None):
		"""Add 1 to the perception."""
		return perception + 1


class InputUpdate(Update):
	"""An update class that uses input."""

	def __call__(self, state, perception, input=None):
		"""Add input to perception if provided."""
		return perception + input if input is not None else perception


@pytest.fixture
def ca() -> CA:
	"""Create a CA instance for testing."""
	perceive = DummyPerceive()
	update = DummyUpdate()
	return CA(perceive, update)


@pytest.fixture
def ca_with_input_update() -> CA:
	"""Create a CA instance with InputUpdate for testing."""
	perceive = DummyPerceive()
	update = InputUpdate()
	return CA(perceive, update)


def test_ca_init(ca: CA) -> None:
	"""Test the initialization of the CA class."""
	assert isinstance(ca.perceive, DummyPerceive)
	assert isinstance(ca.update, DummyUpdate)


def test_ca_step(ca: CA) -> None:
	"""Test the step method of the CA class."""
	state = jnp.array([1.0, 2.0, 3.0])
	new_state, metrics = ca.step(state)
	expected_state = jnp.array([3.0, 5.0, 7.0])

	assert jnp.allclose(new_state, expected_state)
	assert jnp.allclose(metrics, expected_state)


def test_ca_call_single_step(ca: CA) -> None:
	"""Test the CA class with a single step."""
	state = jnp.array([1.0, 2.0, 3.0])
	new_state, metrics = ca(state, num_steps=1)
	expected_state = jnp.array([3.0, 5.0, 7.0])

	assert jnp.allclose(new_state, expected_state)
	assert metrics.shape == (1, 3)  # Metrics should contain the state after the first step
	assert jnp.allclose(metrics[0], expected_state)


def test_ca_call_multiple_steps(ca: CA) -> None:
	"""Test the CA class with multiple steps."""
	state = jnp.array([1.0, 2.0, 3.0])
	final_state, metrics = ca(state, num_steps=3)
	expected_final_state = jnp.array([15.0, 23.0, 31.0])
	expected_metrics = jnp.array([[3.0, 5.0, 7.0], [7.0, 11.0, 15.0], [15.0, 23.0, 31.0]])

	assert jnp.allclose(final_state, expected_final_state)
	assert jnp.allclose(metrics, expected_metrics)


def test_ca_call_metrics_history(ca: CA) -> None:
	"""Test the CA class returns metrics history."""
	state = jnp.array([1.0, 2.0, 3.0])
	_, metrics = ca(state, num_steps=3)
	expected_metrics = jnp.array([[3.0, 5.0, 7.0], [7.0, 11.0, 15.0], [15.0, 23.0, 31.0]])

	assert jnp.allclose(metrics, expected_metrics)


def test_ca_call_with_input(ca_with_input_update: CA) -> None:
	"""Test the CA class with input."""
	state = jnp.array([1.0, 2.0, 3.0])
	input_val = jnp.array([0.1, 0.2, 0.3])
	final_state, metrics = ca_with_input_update(state, input_val, num_steps=1)
	expected_state = jnp.array([2.1, 4.2, 6.3])
	assert jnp.allclose(final_state, expected_state)
	assert jnp.allclose(metrics[0], expected_state)


def test_ca_call_with_input_multiple_steps(ca_with_input_update: CA) -> None:
	"""Test the CA class with input over multiple steps."""
	state = jnp.array([1.0, 2.0, 3.0])
	input_val = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
	final_state, metrics = ca_with_input_update(state, input_val, num_steps=2, input_in_axis=0)
	# Step 1: perceive(state) = [2.0, 4.0, 6.0]
	# Step 1: update = perceive + input[0] = [6.0, 9.0, 12.0] (state_1)
	# Step 2: perceive(state_1) = [12.0, 18.0, 24.0]
	# Step 2: update = perceive + input[1] = [19.0, 26.0, 33.0] (state_2 = final_state)
	expected_final_state = jnp.array([19.0, 26.0, 33.0])
	expected_metrics = jnp.array([[6.0, 9.0, 12.0], [19.0, 26.0, 33.0]])

	assert jnp.allclose(final_state, expected_final_state)
	assert jnp.allclose(metrics, expected_metrics)


def test_ca_vmap(ca: CA) -> None:
	"""Test vectorized mapping of the CA class."""
	states = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	final_states, metrics = nnx.vmap(lambda state: ca(state, num_steps=3))(states)
	expected_final_states = jnp.array([[15.0, 23.0, 31.0], [39.0, 47.0, 55.0]])
	expected_metrics = jnp.array(
		[
			[[3.0, 5.0, 7.0], [7.0, 11.0, 15.0], [15.0, 23.0, 31.0]],  # Batch 1 metrics
			[[9.0, 11.0, 13.0], [19.0, 23.0, 27.0], [39.0, 47.0, 55.0]],  # Batch 2 metrics
		]
	)

	assert jnp.allclose(final_states, expected_final_states)
	assert jnp.allclose(metrics, expected_metrics)
