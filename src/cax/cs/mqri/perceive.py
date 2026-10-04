"""MQRI perception.

Counts live Moore neighbors and the phase sum of those live neighbors so the
update can diffuse phase while applying a majority-life rule.
"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.perceive import Perceive

_SHIFTS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class MqriPerceive(Perceive[Array, Array]):
    """MQRI neighborhood perception.

    Perception channels are ``(alive, neighbor_alive_count, phase, live_phase_sum)``.
    """

    @override
    def __call__(self, state: Array) -> Array:
        """Gather Moore-neighborhood live counts and phase sums.

        Args:
            state: Array ``(..., height, width, 2)`` with alive in channel 0 and
                phase in channel 1.

        Returns:
            Array ``(..., height, width, 4)``.

        """
        alive = state[..., 0]
        phase = state[..., 1]
        neigh = jnp.zeros_like(alive)
        phase_sum = jnp.zeros_like(phase)
        for di, dj in _SHIFTS:
            a = jnp.roll(alive, (di, dj), axis=(-2, -1))
            p = jnp.roll(phase, (di, dj), axis=(-2, -1))
            neigh = neigh + a
            phase_sum = phase_sum + p * a
        return jnp.stack([alive, neigh, phase, phase_sum], axis=-1)
