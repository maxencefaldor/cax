"""Motion module.

Motion time series derived from a center-of-mass trajectory on a torus. Every complex
system emits these under the same names and the same meaning, so a fitness or descriptor
written against `velocity` transfers unchanged from one system to the next.
"""

import jax.numpy as jnp
from jax import Array

# The motion series this module emits
MOTION_SERIES = ("velocity", "linear_velocity", "angle", "angular_velocity")


def motion_series(center_of_mass: Array, *, world_size: Array, T: float) -> dict[str, Array]:
	"""Derive motion time series from a center-of-mass trajectory.

	Args:
		center_of_mass: Trajectory with shape `(num_steps, 2)`, in the system's physical
			length unit.
		world_size: Per-axis extent of the torus in the same unit; displacements are
			unwrapped against it.
		T: Steps per unit time (the reciprocal of the integration time step), so that
			velocities are per unit time rather than per step.

	Returns:
		Mapping of `MOTION_SERIES` names to arrays.

	"""
	# Displacements, unwrapped on the torus
	displacement = jnp.diff(center_of_mass, axis=0)
	displacement = displacement - jnp.round(displacement / world_size) * world_size

	# Instantaneous velocity vector and speed. The vector series matters for fitness:
	# `norm_mean` of `velocity` is the net displacement per unit time, which a spinner
	# (tangent vectors cancelling over a revolution) cannot hack, while the mean of
	# `linear_velocity` rewards any center-of-mass motion, directed or not
	velocity = displacement * T
	linear_velocity = jnp.linalg.norm(displacement, axis=-1) * T

	# Heading direction in [-1, 1] (units of pi) and its rate of change
	angle = jnp.arctan2(displacement[:, 1], displacement[:, 0]) / jnp.pi
	angle_diff = (jnp.diff(angle) + 3) % 2 - 1
	angle_diff = jnp.where(linear_velocity[1:] > 0.01, angle_diff, 0.0)
	angular_velocity = angle_diff * T

	return {
		"velocity": velocity,
		"linear_velocity": linear_velocity,
		"angle": angle,
		"angular_velocity": angular_velocity,
	}
