"""BFF self-replication detector module.

A port of the reference implementation's `CheckSelfRep`. A program is scored by how
reliably it reproduces itself over a chain of executions against random partners: the
score is a byte count in [0, tape_length], and the reference counts a program as a
self-replicator above a threshold (5 in the cubff display, 48 in the 2026 paper).

References:
    [1] BFF: Simple explanations for complex phenomena, Knierim et al. 2026.
        https://arxiv.org/abs/2607.01483
    [2] cubff reference implementation, `CheckSelfRep` in `common_language.h`.
        https://github.com/paradigms-of-intelligence/cubff

"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .interpreter import Control, run

NUM_CHAINS = 13
"""Independent chains per program, each against its own random partner."""

CHAIN_LENGTH = 5
"""Executions per chain; odd so that self-inverting replicators are detected."""


def _run_chains(
    program: Array,
    partners: Array,
    opcode_table: Array,
    *,
    num_steps: int,
    heads_from_tape: bool,
    control: Control,
) -> Array:
    """Run every chain of one program; returns the final tapes, (num_chains, 2L)."""
    length = program.shape[-1]
    num_chains = partners.shape[0]
    tapes = jnp.concatenate(
        [jnp.broadcast_to(program, (num_chains, length)), partners], axis=-1
    )

    def execute(tapes: Array) -> Array:
        return run(
            tapes,
            opcode_table,
            num_steps=num_steps,
            heads_from_tape=heads_from_tape,
            control=control,
            compact_after=num_steps,
        )[0]

    tapes = execute(tapes)
    for _ in range(CHAIN_LENGTH - 1):
        # The child moves to the first half and meets the same partner bytes again.
        tapes = jnp.concatenate([tapes[:, length:], partners], axis=-1)
        tapes = execute(tapes)
    return tapes


def score_from_tapes(program: Array, tapes: Array) -> Array:
    """Score a program from the final tapes of its chains.

    For each byte position, look for a chain whose value there is shared by more than
    `num_chains // 4` chains in total; in the first half the value must also equal the
    program's own byte. Count such positions per half; the score is the smaller count.
    Halves are not compared with each other, so replicators that invert themselves on
    every execution still score.

    Args:
        program: Unsigned 8-bit array of shape (length,).
        tapes: Unsigned 8-bit array of shape (num_chains, 2 * length).

    Returns:
        Scalar int32 score in [0, length].

    """
    length = program.shape[-1]
    num_chains = tapes.shape[0]
    same = tapes[:, None, :] == tapes[None, :, :]  # (a, b, position)
    later = jnp.arange(num_chains)[:, None] < jnp.arange(num_chains)[None, :]
    count = 1 + jnp.sum(same & later[..., None], axis=1)  # (a, position)
    position = jnp.arange(2 * length)
    in_first_half = position < length
    matches_program = tapes[:, :length] == program[None, :]
    valid = jnp.where(
        in_first_half[None, :], jnp.pad(matches_program, ((0, 0), (0, length))), True
    )
    hit = jnp.any(valid & (count > num_chains // 4), axis=0)  # (position,)
    first = jnp.sum(hit[:length])
    second = jnp.sum(hit[length:])
    return jnp.minimum(first, second).astype(jnp.int32)


@partial(jax.jit, static_argnames=("num_steps", "heads_from_tape", "control"))
def replication_score(
    programs: Array,
    partners: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    heads_from_tape: bool = False,
    control: Control = "matched",
) -> Array:
    """Self-replication score of each program in a batch.

    Each program is placed in the first half of a tape whose second half is a random
    partner, executed, and the second half is then moved to the first half and paired
    with the same partner bytes again, `CHAIN_LENGTH` executions in all; this is done
    for `NUM_CHAINS` partners. See `score_from_tapes` for the comparison.

    Args:
        programs: Unsigned 8-bit array of shape (num_programs, length).
        partners: Unsigned 8-bit array of shape (num_programs, NUM_CHAINS, length),
            the partner bytes of each chain; draw them uniformly.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per execution; the reference uses 8192.
        heads_from_tape: Head initialisation convention; see `initial_thread_state`.
        control: Control-flow rule; see the interpreter module.

    Returns:
        Integer array of shape (num_programs,) with scores in [0, length].

    """
    chains = jax.vmap(
        partial(
            _run_chains,
            opcode_table=opcode_table,
            num_steps=num_steps,
            heads_from_tape=heads_from_tape,
            control=control,
        )
    )
    tapes = chains(programs, partners)
    return jax.vmap(score_from_tapes)(programs, tapes)


def sample_partners(key: Array, num_programs: int, length: int = 64) -> Array:
    """Draw the random partner tapes for `replication_score`.

    Args:
        key: PRNG key.
        num_programs: Number of programs to be scored.
        length: Bytes per program.

    Returns:
        Unsigned 8-bit array of shape (num_programs, NUM_CHAINS, length).

    """
    return jax.random.randint(
        key, (num_programs, NUM_CHAINS, length), 0, 256, dtype=jnp.uint8
    )
