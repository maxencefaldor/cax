"""MQRI update.

Majority life plus integer phase diffusion. Frozen cells (fiducial corner and
payload strip) are copied through unchanged so the plate remains decodable.
"""

from typing import override

import jax.numpy as jnp
from jax import Array

from cax.core.update import Update

N_PHASE = 4


class MqriUpdate(Update[Array, Array, Array]):
    """MQRI local update with a frozen mask."""

    def __init__(self, *, freeze: Array):
        """Initialize MQRI update.

        Args:
            freeze: Broadcastable mask ``(..., height, width, 1)`` that is 1.0 on
                cells that must not change (fiducial and payload strip).

        """
        self.freeze = freeze

    @override
    def __call__(
        self, state: Array, perception: Array, input: Array | None = None
    ) -> Array:
        """Advance unfrozen cells; copy frozen cells from ``state``.

        Args:
            state: Current state ``(..., height, width, 2)``.
            perception: Output of ``MqriPerceive``.
            input: Unused.

        Returns:
            Next state with the same shape as ``state``.

        """
        alive = perception[..., 0]
        neigh = perception[..., 1]
        phase = perception[..., 2]
        phase_sum = perception[..., 3]

        new_alive = jnp.logical_or(
            jnp.logical_and(neigh >= 2, neigh <= 4),
            jnp.logical_and(alive >= 1, neigh >= 1),
        ).astype(state.dtype)

        safe_n = jnp.maximum(neigh, 1)
        avg = jnp.where(neigh > 0, jnp.floor(phase_sum / safe_n) % N_PHASE, phase)
        new_phase = jnp.where(new_alive >= 1, avg, 0).astype(state.dtype)

        nxt = jnp.stack([new_alive, new_phase], axis=-1)
        freeze = self.freeze.astype(state.dtype)
        return jnp.where(freeze > 0, state, nxt)
