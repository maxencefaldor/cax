"""MQRI plate automaton.

A two-channel grid (alive, phase) with a frozen fiducial and a frozen payload
strip. The free field follows a majority-life / phase-diffusion rule. Decode
reads the strip — that is the D-gap this system closes. Identity hashes
synthetic Stokes-like I/Q/U of the quantized plate and is not a decoder.
"""

from typing import override

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.utils import clip_and_uint8

from .perceive import MqriPerceive
from .plate import freeze_mask
from .update import MqriUpdate


class Mqri(ComplexSystem[Array, Array]):
    """Mirrored Quartz Reflection Identity plate.

    State shape is ``(height, width, 2)``: channel 0 is alive (0 or 1), channel 1
    is phase in ``{0,1,2,3}``. Bottom ``strip_rows`` and a 3x3 corner are frozen
    so a payload planted in the strip survives rollout.
    """

    def __init__(self, *, height: int = 32, width: int = 32, strip_rows: int = 2):
        """Initialize MQRI.

        Args:
            height: Grid height.
            width: Grid width.
            strip_rows: Frozen bottom rows that hold payload bits.

        """
        freeze = jnp.asarray(freeze_mask(height, width, strip_rows))
        self.height = height
        self.width = width
        self.strip_rows = strip_rows
        self.perceive = MqriPerceive()
        self.update = MqriUpdate(freeze=freeze)

    @override
    def _step(self, state: Array, input: Array | None = None) -> Array:
        perception = self.perceive(state)
        return self.update(state, perception, input)

    @nnx.jit
    @override
    def render(self, state: Array) -> Array:
        """Render alive cells tinted by phase to RGB."""
        alive = state[..., 0:1]
        phase = state[..., 1:2] / 3.0
        rgb = jnp.concatenate(
            [alive * (0.2 + 0.8 * phase), alive * (1.0 - phase), alive],
            axis=-1,
        )
        return clip_and_uint8(rgb)


__all__ = ["Mqri"]
