"""Boids model."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from cax.core.ca import CA, metrics_fn
from cax.types import State
from cax.utils import clip_and_uint8

from .boids_perceive import BoidsPerceive
from .boids_update import BoidsUpdate


class Boids(CA):
	"""Boids model."""

	def __init__(
		self,
		boid_policy: nnx.Module,
		*,
		dt: float = 0.01,
		velocity_half_life: float = jnp.inf,
		boundary: str = "CIRCULAR",
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize Boids."""
		perceive = BoidsPerceive(
			boid_policy=boid_policy,
		)

		update = BoidsUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
			boundary=boundary,
		)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@partial(nnx.jit, static_argnames=("resolution", "boids_size"))
	def render(
		self,
		state: State,
		resolution: int = 512,
		boids_size: float = 0.01,
	) -> jax.Array:
		"""Render state to RGB.

		Args:
			state: An array of states containing position and velocity.
			resolution: Resolution of the output image.
			boids_size: Size of the boids in the [0, 1] coordinate space.

		Returns:
			Rendered states as an RGB image array of shape (resolution, resolution, 3).

		"""
		assert state.position.shape[-1] == 2, "Boids only supports 2D visualization."

		# Adjust coordinates for rendering
		# - Simulation has y increasing upwards (y=0 bottom, y=1 top).
		# - Image has y increasing downwards (y=0 top, y=1 bottom).
		# - Flip position y: map simulation y to image y with (1 - y).
		# - Negate velocity y: adjust direction to match flipped coordinates.
		position = state.position  # Shape: (num_boids, 2)
		position = position.at[:, 1].set(1 - position[:, 1])  # Flip y-coordinate
		velocity = state.velocity  # Shape: (num_boids, 2)
		velocity = velocity.at[:, 1].set(-velocity[:, 1])  # Negate y-component

		# Compute unit velocity and perpendicular vectors
		v_norm = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
		v_hat = velocity / (v_norm + 1e-8)
		v_perp = jnp.stack([-v_hat[..., 1], v_hat[..., 0]], axis=-1)

		# Define triangle dimensions
		h = 2 * boids_size  # Height from base to tip
		w = boids_size  # Base width

		# Compute triangle vertices
		vertex0 = position - (w / 2) * v_perp  # Base left
		vertex1 = position + h * v_hat  # Tip
		vertex2 = position + (w / 2) * v_perp  # Base right
		vertices = jnp.stack([vertex0, vertex1, vertex2], axis=1)  # Shape: (num_boids, 3, 2)

		# Create grid of pixel centers
		x = jnp.linspace(0, 1, resolution)
		y = jnp.linspace(0, 1, resolution)
		grid = jnp.stack(jnp.meshgrid(x, y), axis=-1)  # Shape: (resolution, resolution, 2)

		# Compute squared distances to all boids
		distance_sq = jnp.sum((grid[:, :, None, :] - position[None, None, :, :]) ** 2, axis=-1)
		# Shape: (resolution, resolution, num_boids)

		# Find index of closest boid
		closest_idx = jnp.argmin(distance_sq, axis=-1)  # Shape: (resolution, resolution)

		# Get vertices of the closest boid
		closest_vertices = vertices[closest_idx, :, :]  # Shape: (resolution, resolution, 3, 2)

		# Extract vertices
		a = closest_vertices[..., 0, :]  # vertex0
		b = closest_vertices[..., 1, :]  # vertex1
		c = closest_vertices[..., 2, :]  # vertex2

		# Compute edge vectors
		edge0 = b - a
		edge1 = c - b
		edge2 = a - c

		# Compute cross products for point-in-triangle test
		q = grid
		cross0 = edge0[..., 0] * (q - a)[..., 1] - edge0[..., 1] * (q - a)[..., 0]
		cross1 = edge1[..., 0] * (q - b)[..., 1] - edge1[..., 1] * (q - b)[..., 0]
		cross2 = edge2[..., 0] * (q - c)[..., 1] - edge2[..., 1] * (q - c)[..., 0]

		# Determine if pixel is inside the triangle
		inside = (cross0 > 0) & (cross1 > 0) & (cross2 > 0)  # Shape: (resolution, resolution)

		# Create RGB image
		gray = inside[..., None].astype(jnp.float32)  # Shape: (resolution, resolution, 1)
		rgb = jnp.repeat(gray, 3, axis=-1)  # Shape: (resolution, resolution, 3)

		return clip_and_uint8(rgb)
