"""Module for handling state transformations in the CAX framework."""

import jax.numpy as jnp

from cax.types import State


def state_to_alive(state: State) -> State:
	"""Extract the 'alive' component from the state.

	Args:
		state: The input state.

	Returns:
		State: The 'alive' component of the state.

	"""
	return state[..., -1:]


def state_to_rgba(state: State) -> State:
	"""Extract the RGBA components from the state.

	Args:
		state: The input state.

	Returns:
		State: The RGBA components of the state.

	"""
	return state[..., -4:]


def state_from_rgba_to_rgb(state: State) -> State:
	"""Convert RGBA state to RGB state.

	Args:
		state: The input state in RGBA format.

	Returns:
		State: The converted RGB state.

	"""
	rgba = state_to_rgba(state)
	rgb, alive = rgba[..., :-1], rgba[..., -1:]
	alpha = jnp.clip(alive, min=0.0, max=1.0)
	return 1.0 - alpha + rgb  # (1.0 - alpha) * 1.0 + alpha * rgb, assume rgb has been multiplied by alpha
