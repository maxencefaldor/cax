"""Smoothed particle hydrodynamics perceive module.

Where a convolution reads a fixed stencil of neighbors, particles have no stencil: who
is nearby changes every step. Smoothed particle hydrodynamics answers the same questions
--- what is the field here, which way is it changing --- as sums over whatever particles
happen to lie within a support radius, weighted by a kernel that falls smoothly to zero
at the edge. The smoothness is what keeps the whole thing differentiable as neighbors
come and go.
"""

import math
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ._sph_kernel import sph_moments
from .perceive import Perceive


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Particles:
    """A cloud of particles, each carrying a position and an internal state.

    Attributes:
        position: Positions with shape `(..., num_particles, num_spatial_dims)`.
        state: Internal states with shape `(..., num_particles, channel_size)`.

    """

    position: Array
    state: Array


def poly6_kernel(square_distance: Array, support_radius: float) -> Array:
    """Evaluate the Poly6 smoothing kernel from a squared distance.

    Poly6 takes the squared distance directly, so nothing has to take a square root to
    use it, and it falls to zero smoothly at the support radius.

    Args:
        square_distance: Squared distances between pairs of particles.
        support_radius: Distance beyond which particles do not interact.

    Returns:
        Kernel weights, zero outside the support radius.

    """
    normalization = 4.0 / (math.pi * support_radius**8)
    inside = support_radius**2 - square_distance
    return jnp.where(inside > 0.0, normalization * inside**3, 0.0)


def spiky_gradient_kernel(
    displacement: Array, distance: Array, support_radius: float
) -> Array:
    """Evaluate the Spiky kernel's gradient.

    The gradient of Poly6 vanishes as two particles approach each other, which leaves
    close neighbors --- the ones that should matter most --- contributing nothing to a
    gradient estimate. Spiky's does not, so it is the one used wherever a direction is
    needed.

    Args:
        displacement: Vectors between pairs of particles, with shape
            `(..., num_spatial_dims)`.
        distance: Their lengths.
        support_radius: Distance beyond which particles do not interact.

    Returns:
        Kernel gradients with the same shape as `displacement`.

    """
    normalization = 30.0 / (math.pi * support_radius**5)
    inside = support_radius - distance
    safe_distance = jnp.maximum(distance, 1e-9)
    # Negative: the kernel falls off with distance, so its gradient points inward.
    scale = jnp.where(
        (inside > 0.0) & (distance > 1e-9),
        -normalization * inside**2 / safe_distance,
        0.0,
    )
    return displacement * scale[..., None]


def _log_normalize(vector: Array) -> Array:
    """Keep a vector's direction, and compress its length to `log(1 + length)`.

    The zero vector needs care rather than a clamp. A length is a square root, whose
    derivative is unbounded at the origin, so differentiating through one there gives
    `nan` however small the epsilon underneath it --- and most particles sit in a
    neighborhood where nothing is happening yet, so the vector really is zero. Branching
    before the square root, not after, keeps the derivative finite: the constant the
    masked branch takes has no gradient at all.
    """
    square_length = jnp.sum(jnp.square(vector), axis=-1, keepdims=True)
    positive = square_length > 0.0
    length = jnp.sqrt(jnp.where(positive, square_length, 1.0))
    return vector * jnp.where(positive, jnp.log1p(length) / length, 1.0)


class SPHPerceive(Perceive[Particles, Array]):
    """Perceive a particle's neighborhood with smoothed particle hydrodynamics.

    Four quantities are gathered for every particle: its own state, the
    neighborhood-average state, the gradient of the state, and the gradient of the
    particle density.

    The average stands in for a Laplacian. The difference between a particle's own state
    and its neighborhood average is proportional to the Laplacian for small radii, and
    the explicit smoothed particle hydrodynamics Laplacian is numerically ill-behaved,
    so the average is carried instead and the network can take the difference itself.

    The density gradient has no counterpart on a lattice, where cells are evenly spaced
    by construction. Particles crowd and spread, so it is the signal that says which way
    the cloud is getting denser.

    Every pair is evaluated. That is quadratic in the number of particles, and for the
    sizes these systems are trained at it is also the fastest thing to do: the
    neighborhood holds a few percent of the cloud, so a spatial index spends more on
    bookkeeping than it saves on arithmetic.
    """

    def __init__(
        self,
        *,
        support_radius: float,
        mass: float = 1.0,
        log_normalize: bool = True,
        period: float | None = None,
        fused: bool = False,
    ):
        """Initialize SPH perceive.

        Args:
            support_radius: Distance beyond which particles do not interact.
            mass: What each particle weighs. Density is a sum of masses, so this sets
                the scale of the density gradient --- and that quantity is handed to a
                network alongside states of order one. A cloud of `n` particles usually
                wants `1 / n`, which makes the whole cloud weigh one and leaves the
                density near one as well.
            log_normalize: Whether to compress the two gradients to `log(1 + norm)`
                while keeping their direction. A kernel gradient carries a factor of the
                support radius to a negative power, so where particles crowd it reaches
                thousands while the states beside it are of order one --- and a network
                fed both diverges. Compressing keeps the direction, which is the part
                that means something, and puts the magnitude on a scale the rest of the
                perception shares.
            period: Size of the periodic domain, or None for an unbounded one.
            fused: Whether to compute the sums with Pallas kernels instead of array
                operations. The two agree; the kernels never build anything with an
                entry per pair, so they run faster and reach cloud sizes the array
                version cannot fit. Any particle count works --- the tile shape is
                chosen from it, and a count the tiles do not divide is padded and
                masked. GPU-only, and written for two spatial dimensions.

        """
        self.support_radius = support_radius
        self.mass = mass
        self.log_normalize = log_normalize
        self.period = period
        self.fused = fused

    @override
    def __call__(self, state: Particles) -> Array:
        """Perceive the neighborhood of every particle.

        Args:
            state: The cloud to perceive.

        Returns:
            Perception with shape `(..., num_particles, perception_size)`, holding the
                state, the neighborhood average, the state gradient and the density
                gradient, concatenated in that order.

        """
        if self.fused:
            return self._fused(state)

        position, feature = state.position, state.state
        num_spatial_dims = position.shape[-1]

        displacement = position[..., :, None, :] - position[..., None, :, :]
        if self.period is not None:
            displacement -= self.period * jnp.round(displacement / self.period)

        square_distance = jnp.sum(jnp.square(displacement), axis=-1)
        distance = jnp.sqrt(jnp.maximum(square_distance, 1e-18))

        weight = poly6_kernel(square_distance, self.support_radius)
        weight_gradient = spiky_gradient_kernel(
            displacement, distance, self.support_radius
        )

        density = self.mass * jnp.sum(weight, axis=-1)

        # Each neighbor is weighted by the volume *it* stands for, not by the volume of
        # the particle doing the looking. The two agree while the cloud is even and part
        # ways once it is not, which is the whole of what a growing system does.
        volume = self.mass / jnp.maximum(density, 1e-8)

        average = jnp.einsum("...ij,...j,...jc->...ic", weight, volume, feature)

        # The difference form: what the neighbors' states are *relative to this one*.
        # Taking the states directly would leave a constant field with a non-zero
        # gradient wherever the cloud is uneven, since the kernel gradients would no
        # longer cancel.
        difference = feature[..., None, :, :] - feature[..., :, None, :]
        gradient = jnp.einsum(
            "...ijd,...j,...ijc->...icd", weight_gradient, volume, difference
        )
        density_gradient = self.mass * jnp.sum(weight_gradient, axis=-2)

        if self.log_normalize:
            gradient = _log_normalize(gradient)
            density_gradient = _log_normalize(density_gradient)

        return jnp.concatenate(
            [
                feature,
                average,
                gradient.reshape(
                    *feature.shape[:-1], feature.shape[-1] * num_spatial_dims
                ),
                density_gradient,
            ],
            axis=-1,
        )

    def _fused(self, state: Particles) -> Array:
        """Perceive through the Pallas kernels, which take one cloud at a time."""
        position, feature = state.position, state.state

        if position.ndim > 2:
            return jax.vmap(self._fused)(state)

        num_spatial_dims = position.shape[-1]
        if num_spatial_dims != 2:
            raise ValueError(
                "fused=True is written for two spatial dimensions, "
                f"got {num_spatial_dims}."
            )
        _, average, gradient, density_gradient = sph_moments(
            position, feature, self.support_radius, self.mass, self.period
        )

        if self.log_normalize:
            gradient = _log_normalize(gradient)
            density_gradient = _log_normalize(density_gradient)

        return jnp.concatenate(
            [
                feature,
                average,
                gradient.reshape(
                    *feature.shape[:-1], feature.shape[-1] * num_spatial_dims
                ),
                density_gradient,
            ],
            axis=-1,
        )

    @staticmethod
    def perception_size(*, channel_size: int, num_spatial_dims: int) -> int:
        """Return the width of the perception this produces.

        Args:
            channel_size: Number of state channels per particle.
            num_spatial_dims: Number of spatial dimensions.

        Returns:
            The perception width.

        """
        return channel_size * (2 + num_spatial_dims) + num_spatial_dims
