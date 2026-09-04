"""Checkpoint module.

Asynchronous orbax checkpointing of the full evolution state — DNS state, encoder
parameters, and rng key. Runs are cheap and are never resumed today; the checkpoints
are still written so resuming could be built back if that changes.
"""

from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array

from .dns import DNSState
from .encoder import Encoder


def checkpoint_manager(directory: Path) -> ocp.CheckpointManager:
    """Create an async checkpoint manager for a run's checkpoint directory."""
    return ocp.CheckpointManager(
        directory.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2, enable_async_checkpointing=True
        ),
    )


def save(
    manager: ocp.CheckpointManager,
    generation: int,
    state: DNSState,
    encoder_fn: Encoder,
    key: Array,
) -> None:
    """Save the evolution state at a generation (asynchronously)."""
    items = {
        "state": ocp.args.StandardSave(state),
        "key": ocp.args.StandardSave({"data": jax.random.key_data(key)}),
    }
    params = nnx.state(encoder_fn, nnx.Param).to_pure_dict()
    if params:
        items["encoder"] = ocp.args.StandardSave(params)
    manager.save(generation, args=ocp.args.Composite(**items))
