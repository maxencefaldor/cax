"""Swarm 1-D tag consensus.

A line of cells votes for local tags (idle / lit / run / match). Isolated
workers cannot see across a chunk cut; chained workers can. This is a
hand-coded rule, not a trained NCA.
"""

from typing import override

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8

from .perceive import SwarmPerceive
from .update import SwarmUpdate

TAG_COLORS = jnp.array(
    [
        [0.0, 0.0, 0.0],  # idle
        [0.7, 0.7, 0.7],  # lit
        [0.2, 0.6, 1.0],  # run
        [1.0, 0.5, 0.1],  # match
    ]
)


class Swarm(ComplexSystem[Array, Array]):
    """1-D swarm of tag-voting workers.

    State shape is ``(length, 1)`` with tags in ``{0,1,2,3}``.
    ``worker_size`` is the isolated-worker width. ``isolated=True`` zeros
    neighbors that sit on the other side of a worker boundary.
    """

    def __init__(self, *, worker_size: int = 16, isolated: bool = True):
        """Initialize swarm.

        Args:
            worker_size: Cells per worker.
            isolated: If True, workers do not read across the chunk cut.

        """
        self.worker_size = worker_size
        self.isolated = isolated
        self.perceive = SwarmPerceive(worker_size=worker_size, isolated=isolated)
        self.update = SwarmUpdate()

    @override
    def _step(self, state: Array, input: Array | None = None) -> Array:
        perception = self.perceive(state)
        return self.update(state, perception, input)

    @nnx.jit
    @override
    def render(self, state: Array) -> Array:
        """Map tag ids to RGB along a 1-D strip (height 1)."""
        tags = jnp.clip(jnp.round(state[..., 0]).astype(jnp.int32), 0, 3)
        rgb = TAG_COLORS[tags]
        return clip_and_uint8(rgb)
