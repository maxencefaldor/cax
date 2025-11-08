"""Boids module.

This module implements Boids, a particle-based system that simulates the flocking behavior
of birds or fish through simple local rules. Each boid (bird-like object) adjusts its
velocity based on the boid policy. These local interactions produce emergent global flocking
patterns without centralized coordination.
"""

from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input
from cax.utils import clip_and_uint8

from .perceive import BoidsPerceive
from .policy import BoidPolicy
from .state import BoidsState
from .update import BoidsUpdate


class Boids(ComplexSystem):
	"""Boids class."""

	def __init__(
		self,
		*,
		dt: float = 0.01,
		velocity_half_life: float = jnp.inf,
		boid_policy: BoidPolicy,
	):
		"""Initialize Boids.

		Args:
			dt: Time step of the simulation in arbitrary time units. Smaller values
				produce smoother motion but require more steps for the same duration.
			velocity_half_life: Time constant for velocity decay due to friction. After
				this time, velocity is halved without steering input. Use jnp.inf for no
				friction. Smaller values create more damped, sluggish motion.
			boid_policy: Policy defining the behavior of the boids.

		"""
		self.perceive = BoidsPerceive(
			boid_policy=boid_policy,
		)
		self.update = BoidsUpdate(
			dt=dt,
			velocity_half_life=velocity_half_life,
		)

	def _step(
		self, state: BoidsState, input: Input | None = None, *, sow: bool = False
	) -> BoidsState:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@partial(nnx.jit, static_argnames=("resolution", "boids_size"))
	def render(
		self,
		state: BoidsState,
		*,
		resolution: int = 512,
		boids_size: float = 0.01,
	) -> Array:
		"""Render state to RGB image.

		Renders boids as triangular agents pointing in their direction of motion on a
		white background. Each boid is drawn as a filled triangle with the tip pointing
		in the velocity direction, providing visual feedback about both position and
		heading. The visualization uses 2D coordinates in the range [0, 1].

		Args:
			state: BoidsState containing position and velocity arrays. Position should have
				shape (num_boids, 2) with coordinates in [0, 1]. Velocity determines the
				orientation and can have arbitrary magnitude.
			resolution: Size of the output image in pixels for both width and height.
				Higher values produce smoother, more detailed renderings.
			boids_size: Base width of each boid triangle in coordinate space [0, 1].
				The triangle height is twice this value. Larger values make boids more
				visible but may cause overlap.

		Returns:
			RGB image with dtype uint8 and shape (resolution, resolution, 3), where
				boids appear as black triangles on a white background.

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
