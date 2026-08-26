"""Shared dynamics primitives for particle systems on the unit torus.

Boids and Particle Life integrate the same way: exponential velocity damping, a
semi-implicit Euler step, and periodic boundary conditions. These helpers hold that
shared physics in one place; each system keeps its own defaults and state types.
"""

import jax.numpy as jnp
from jax import Array


def toroidal_difference(position_1: Array, position_2: Array, *, period: float = 1.0) -> Array:
	"""Minimum-image vector from `position_1` to `position_2` on a torus.

	Applies periodic boundary conditions component-wise so each component of the
	result lies in `[-period / 2, period / 2]` — the shortest displacement on a
	torus of the given period.

	Args:
		position_1: Start positions.
		position_2: End positions, broadcastable against `position_1`.
		period: Length of the torus along every axis.

	Returns:
		Component-wise shortest displacement from `position_1` to `position_2`.

	"""
	difference = position_2 - position_1
	difference = jnp.where(difference > period / 2, difference - period, difference)
	difference = jnp.where(difference < -period / 2, difference + period, difference)
	return difference


def damped_euler_step(
	position: Array,
	velocity: Array,
	acceleration: Array,
	*,
	dt: float,
	friction_factor: float,
	period: float = 1.0,
) -> tuple[Array, Array]:
	"""Semi-implicit Euler step with velocity damping and periodic boundaries.

	The velocity is damped by `friction_factor` (typically `0.5 ** (dt / half_life)`)
	and accelerated, then the position is advanced with the *new* velocity and wrapped
	onto the torus.

	Args:
		position: Positions on the torus.
		velocity: Velocities.
		acceleration: Accelerations from the perception step.
		dt: Time step.
		friction_factor: Multiplicative velocity decay per step.
		period: Length of the torus along every axis.

	Returns:
		A `(position, velocity)` tuple after one step.

	"""
	velocity = friction_factor * velocity + dt * acceleration
	position = (position + dt * velocity) % period
	return position, velocity
