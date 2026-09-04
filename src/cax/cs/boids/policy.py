"""Boid policy following Craig W. Reynolds (1987).

References:
    [1] Craig W. Reynolds. 1987. Flocks, herds and schools: A distributed behavioral
        model.
    [2] https://www.red3d.com/cwr/papers/1987/boids.html

"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.utils import safe_divide, safe_norm, toroidal_difference

from .state import BoidsState


class BoidsPolicy(nnx.Module):
    """Boid policy according to Craig Reynolds' paper.

    The three behavior weights are `nnx.Param` on purpose: they are the knobs a
    differentiable-flocking experiment trains. Everything else is fixed configuration.

    Simulation noise is drawn from the `noise` RNG stream, so callers can seed it
    independently of initialization (`nnx.Rngs(0)` works too — every stream falls back
    to the default seed). The policy carries its `Rngs`, which makes it stateful: two
    identical calls draw different noise, and reproducibility runs through the seed.
    """

    def __init__(
        self,
        *,
        acceleration_max: float = jnp.inf,
        acceleration_scale: float = 1.0,
        perception_radius: float = 0.2,
        separation_distance: float = 0.02,
        separation_weight: float = 4.5,
        alignment_weight: float = 0.65,
        cohesion_weight: float = 0.75,
        noise_scale: float = 0.05,
        rngs: nnx.Rngs,
    ):
        """Initialize boid policy.

        Args:
            acceleration_max: Maximum acceleration.
            acceleration_scale: Scale for acceleration.
            perception_radius: Radius within which other boids are perceived.
            separation_distance: Separation distance.
            separation_weight: Weight for separation force.
            alignment_weight: Weight for alignment force.
            cohesion_weight: Weight for cohesion force.
            noise_scale: Scale for noise.
            rngs: rng key.

        """
        self.acceleration_max = acceleration_max
        self.acceleration_scale = acceleration_scale
        self.perception_radius = perception_radius
        self.separation_distance = separation_distance

        self.separation_weight = nnx.Param(separation_weight)
        self.alignment_weight = nnx.Param(alignment_weight)
        self.cohesion_weight = nnx.Param(cohesion_weight)
        self.noise_scale = nnx.Param(noise_scale)

        self.rngs = rngs

    def _toroidal_distance2(self, position_1: Array, position_2: Array) -> Array:
        """Calculate squared distance considering toroidal world in [0, 1]^n."""
        vector = toroidal_difference(position_1, position_2)
        return jnp.sum(vector**2)

    def _normalize(self, vector: Array) -> Array:
        """Normalize a vector, returning zero for the zero vector."""
        norm = safe_norm(vector)
        return safe_divide(vector, norm, where=norm > 0.0)

    def _clip_by_norm(self, vector: Array, max_val: float) -> Array:
        """Limit the magnitude of a vector."""
        norm = safe_norm(vector)
        is_over = norm > max_val
        # Double-where: the divisor is sanitized before dividing, so the untaken branch
        # cannot leak nan into the gradient (see cax.utils.numerics).
        norm_safe = jnp.where(is_over, norm, jnp.ones_like(norm))
        return jnp.where(is_over, vector * max_val / norm_safe, vector)

    def _masked_mean(self, values: Array, mask: Array) -> Array:
        """Mean of `values` over rows where `mask` is true; zero when no row is.

        `jnp.mean(..., where=mask)` divides by the count and returns nan for an empty
        mask — a boid with no neighbor would poison the whole flock. Sum and divide
        safely instead.

        Args:
            values: Array with shape `(num_boids, num_spatial_dims)`.
            mask: Boolean array with shape `(num_boids,)`.

        Returns:
            Mean over the selected rows, or a zero vector when none is selected.

        """
        total = jnp.sum(values, axis=0, where=mask[..., None])
        count = jnp.sum(mask)
        return safe_divide(total, count, where=count > 0)

    def separation(self, state: BoidsState, boid_idx: int) -> Array:
        """Calculate separation force for a boid."""
        # Calculate distances to all other boids
        distances = nnx.vmap(
            lambda position: self._toroidal_distance2(
                state.position[boid_idx], position
            )
        )(state.position)

        # Create masks for filtering
        is_self = jnp.arange(len(state.position)) == boid_idx
        is_too_close = distances <= self.separation_distance**2

        # Only consider other boids that are too close
        separation_mask = ~is_self & is_too_close

        # Calculate steering force
        separations = -toroidal_difference(state.position[boid_idx], state.position)
        steer = jnp.sum(separations, axis=0, where=separation_mask[..., None])

        return self._normalize(steer)

    def alignment(self, state: BoidsState, boid_idx: int) -> Array:
        """Calculate alignment force for a boid."""
        # Calculate distances to all other boids
        distances = nnx.vmap(
            lambda position: self._toroidal_distance2(
                state.position[boid_idx], position
            )
        )(state.position)

        # Create masks for filtering
        is_self = jnp.arange(len(state.position)) == boid_idx
        is_in_perception = distances <= self.perception_radius**2

        # Only consider other boids within perception radius
        perception_mask = ~is_self & is_in_perception

        # Calculate steering force
        velocity_avg = self._masked_mean(state.velocity, perception_mask)
        steer = velocity_avg - state.velocity[boid_idx]

        return self._normalize(steer)

    def cohesion(self, state: BoidsState, boid_idx: int) -> Array:
        """Calculate cohesion force for a boid."""
        # Calculate distances to all other boids
        distances = nnx.vmap(
            lambda position: self._toroidal_distance2(
                state.position[boid_idx], position
            )
        )(state.position)

        # Create masks for filtering
        is_self = jnp.arange(len(state.position)) == boid_idx
        is_in_perception = distances <= self.perception_radius**2

        # Only consider other boids within perception radius
        perception_mask = ~is_self & is_in_perception

        # Calculate steering force
        position_avg = nnx.vmap(
            lambda position: toroidal_difference(state.position[boid_idx], position)
        )(state.position)
        steer = self._masked_mean(position_avg, perception_mask)

        return self._normalize(steer)

    def __call__(self, state: BoidsState, boid_idx: int) -> Array:
        """Apply the boid policy.

        Args:
            state: Position and velocity of all boids.
            boid_idx: Index of the current boid.

        Returns:
            Acceleration of the current boid.

        """
        # Apply each rule, get resulting forces, and weight them
        separation_update = self.separation_weight * self.separation(state, boid_idx)
        alignment_update = self.alignment_weight * self.alignment(state, boid_idx)
        cohesion_update = self.cohesion_weight * self.cohesion(state, boid_idx)

        # Combine forces
        acceleration = separation_update + alignment_update + cohesion_update

        # Scale and add noise
        acceleration *= self.acceleration_scale
        acceleration += self.noise_scale * jax.random.uniform(
            self.rngs.noise(),
            shape=acceleration.shape,
            minval=-1.0,
            maxval=1.0,
        )

        # Limit acceleration
        acceleration = self._clip_by_norm(acceleration, self.acceleration_max)

        return acceleration
