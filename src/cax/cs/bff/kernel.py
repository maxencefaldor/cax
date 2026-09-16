"""One-kernel BFF interpreter for GPUs.

`run_kernel` computes the same function as `interpreter.run`, but inside a single Pallas
kernel: each block of tapes loops over the whole step budget on-chip and returns as soon
as every tape in the block has halted. The XLA version pays 20 to 40 kernel launches
per step for every tape, halted or not, which is a floor of about 40 µs per step on an
H100 whatever the batch size; here a step of a block is a few dozen instructions, and a
block of random tapes typically exits after a few hundred steps.

The bracket search is a sequential walk from the bracket, exactly the reference's loop,
vectorised over the block and stopped as soon as every searching tape has its answer
or has left the tape. It runs only on steps where some tape in the block takes a jump.

The kernel targets the Triton backend and runs under `interpret=True` on any backend for
tests. Tapes are updated in place through an input-output alias.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .interpreter import Control, initial_thread_state
from .language import Op, is_instruction


def _kernel(
    table_ref,
    tape_in_ref,
    pc_ref,
    head0_ref,
    head1_ref,
    tape_ref,
    steps_ref,
    ops_ref,
    *,
    num_steps: int,
    length: int,
    block: int,
    control: Control,
) -> None:
    del tape_in_ref  # aliased to tape_ref
    rows = jnp.arange(block, dtype=jnp.int32)
    zero = jnp.zeros((block,), dtype=jnp.int32)

    def search(pc: Array, forward: Array, need: Array) -> Array:
        """Position of the matching bracket for the tapes in `need`, else -1."""
        sign = jnp.where(forward, 1, -1).astype(jnp.int32)

        def cond(carry):
            k, _depth, found, _target = carry
            active = need & ~found
            if control == "matched":
                pos = pc + sign * k
                active = active & (pos >= 0) & (pos < length)
            return (k < length) & (jnp.sum(active.astype(jnp.int32)) > 0)

        def body(carry):
            k, depth, found, target = carry
            pos = pc + sign * k
            if control == "matched":
                inside = (pos >= 0) & (pos < length)
                pos = jnp.clip(pos, 0, length - 1)
            else:
                inside = jnp.ones_like(need)
                pos = (pos + length) % length
            op = table_ref[tape_ref[rows, pos].astype(jnp.int32)]
            opening = (op == Op.LOOP_START).astype(jnp.int32)
            closing = (op == Op.LOOP_END).astype(jnp.int32)
            delta = jnp.where(forward, opening - closing, closing - opening)
            active = need & ~found & inside
            depth = jnp.where(active, depth + delta, depth)
            hit = active & (depth == 0)
            target = jnp.where(hit, pos, target)
            return k + 1, depth, found | hit, target

        init = (jnp.int32(1), zero + 1, jnp.zeros_like(need), zero - 1)
        _k, _depth, _found, target = jax.lax.while_loop(cond, body, init)
        return target

    def step_cond(carry):
        i, _pc, _h0, _h1, _direction, live, _steps, _ops = carry
        return (i < num_steps) & (jnp.sum(live.astype(jnp.int32)) > 0)

    def step_body(carry):
        i, pc, head0, head1, direction, live, steps, ops = carry
        cmd = table_ref[tape_ref[rows, pc].astype(jnp.int32)]
        value0 = tape_ref[rows, head0]
        value1 = tape_ref[rows, head1]
        jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
        jump_backward = (cmd == Op.LOOP_END) & (value0 != 0)

        # Writes: at most one byte changes, at head0 or head1.
        write_pos = jnp.where(cmd == Op.COPY01, head1, head0)
        current = tape_ref[rows, write_pos]
        write_val = jnp.where(
            cmd == Op.PLUS,
            value0 + 1,
            jnp.where(
                cmd == Op.MINUS,
                value0 - 1,
                jnp.where(
                    cmd == Op.COPY01,
                    value0,
                    jnp.where(cmd == Op.COPY10, value1, current),
                ),
            ),
        ).astype(jnp.uint8)
        tape_ref[rows, write_pos] = jnp.where(live, write_val, current)

        # Head moves, wrapping on the tape; `~` exchanges the heads.
        inc0 = (cmd == Op.INC0).astype(jnp.int32) - (cmd == Op.DEC0).astype(jnp.int32)
        inc1 = (cmd == Op.INC1).astype(jnp.int32) - (cmd == Op.DEC1).astype(jnp.int32)
        moved0 = (head0 + inc0 + length) % length
        moved1 = (head1 + inc1 + length) % length
        swap = cmd == Op.SWAP
        new_head0 = jnp.where(swap, head1, moved0)
        new_head1 = jnp.where(swap, head0, moved1)

        # Control flow.
        jumping = jump_forward | jump_backward
        if control == "flip":
            new_direction = jnp.where(jumping, -direction, direction)
            new_pc = (pc + new_direction + length) % length
            new_halted = jnp.zeros_like(live)
        else:
            new_direction = direction
            target = search(pc, jump_forward, live & jumping)
            found = target >= 0
            jumped = jnp.where(found, target, pc)
            if control == "matched":
                jumped = jnp.where(
                    jump_forward & ~found,
                    length,
                    jnp.where(jump_backward & ~found, -2, jumped),
                )
            new_pc = jnp.where(jumping, jumped, pc) + 1
            if control == "matched":
                new_halted = (new_pc < 0) | (new_pc >= length)
                new_pc = jnp.clip(new_pc, 0, length - 1)
            else:
                new_pc = new_pc % length
                new_halted = jnp.zeros_like(live)

        return (
            i + 1,
            jnp.where(live, new_pc, pc),
            jnp.where(live, new_head0, head0),
            jnp.where(live, new_head1, head1),
            jnp.where(live, new_direction, direction),
            live & ~new_halted,
            steps + live.astype(jnp.int32),
            ops + (live & is_instruction(cmd)).astype(jnp.int32),
        )

    init = (
        jnp.int32(0),
        pc_ref[...],
        head0_ref[...],
        head1_ref[...],
        zero + 1,
        jnp.ones((block,), dtype=bool),
        zero,
        zero,
    )
    _i, _pc, _h0, _h1, _d, _live, steps, ops = jax.lax.while_loop(
        step_cond, step_body, init
    )
    steps_ref[...] = steps
    ops_ref[...] = ops


@partial(
    jax.jit,
    static_argnames=("num_steps", "heads_from_tape", "control", "block", "interpret"),
)
def run_kernel(
    tapes: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    heads_from_tape: bool = False,
    control: Control = "matched",
    block: int = 32,
    interpret: bool = False,
) -> tuple[Array, Array, Array]:
    """Run a batch of tapes inside one kernel; same function as `interpreter.run`.

    Args:
        tapes: Unsigned 8-bit array of shape (num_tapes, length).
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per tape.
        heads_from_tape: Whether the first two bytes seed the heads.
        control: Control-flow rule; see the interpreter module.
        block: Tapes per kernel block; a block runs until its last tape halts, so
            smaller blocks waste fewer lanes on random soups. One warp is 32.
        interpret: Run the kernel in Pallas interpret mode (any backend, slow).

    Returns:
        A tuple `(tapes, steps, ops)` as `interpreter.run`.

    """
    num, length = tapes.shape
    padded = -(-num // block) * block
    if padded != num:
        tapes = jnp.concatenate(
            [tapes, jnp.zeros((padded - num, length), dtype=tapes.dtype)]
        )
    init = jax.vmap(partial(initial_thread_state, heads_from_tape=heads_from_tape))
    state = init(tapes)
    kernel = partial(
        _kernel, num_steps=num_steps, length=length, block=block, control=control
    )
    tape_spec = pl.BlockSpec((block, length), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((block,), lambda i: (i,))
    vec_shape = jax.ShapeDtypeStruct((padded,), jnp.int32)
    out_tapes, steps, ops = pl.pallas_call(
        kernel,
        grid=(padded // block,),
        in_specs=[
            pl.BlockSpec((256,), lambda i: (0,)),
            tape_spec,
            vec_spec,
            vec_spec,
            vec_spec,
        ],
        out_specs=(tape_spec, vec_spec, vec_spec),
        out_shape=(
            jax.ShapeDtypeStruct((padded, length), tapes.dtype),
            vec_shape,
            vec_shape,
        ),
        input_output_aliases={1: 0},
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=max(1, block // 32)),
    )(
        opcode_table.astype(jnp.int32),
        tapes,
        state.pc.astype(jnp.int32),
        state.head0.astype(jnp.int32),
        state.head1.astype(jnp.int32),
    )
    return out_tapes[:num], steps[:num], ops[:num]


__all__ = ["run_kernel"]
