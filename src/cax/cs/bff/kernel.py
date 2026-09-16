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

from .interpreter import Control
from .language import Op, is_instruction


def _kernel(
    table_ref,
    brackets_ref,
    tape_in_ref,
    tape_ref,
    steps_ref,
    ops_ref,
    *,
    num: int,
    padded: int,
    heads_from_tape: bool,
    num_steps: int,
    length: int,
    block: int,
    control: Control,
    search_chunk: int,
    unroll: int,
) -> None:
    del tape_in_ref  # aliased to tape_ref
    # Whole-array refs addressed by row. The tape has `num` real rows, spare rows
    # for the last block's extra lanes, and one dummy row at the end that every
    # dead lane points at: a dead lane's masked accesses then never share an
    # address with a live lane's, which the interpreter's read-modify-write
    # discharge of masked stores would otherwise get wrong.
    lanes = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = lanes < num
    dummy = padded - 1
    rows = jnp.where(valid, lanes, dummy)  # for the loads and stores outside the loop
    zero = jnp.zeros((block,), dtype=jnp.int32)
    open_byte = brackets_ref[0]
    close_byte = brackets_ref[1]

    def search(rows: Array, pc: Array, forward: Array, need: Array) -> Array:
        """Position of the matching bracket for the tapes in `need`, else -1."""
        sign = jnp.where(forward, 1, -1).astype(jnp.int32)

        def cond(carry):
            k, _depth, found, _target = carry
            active = need & ~found
            if control == "matched":
                pos = pc + sign * k
                active = active & (pos >= 0) & (pos < length)
            return (k < length) & (jnp.sum(active.astype(jnp.int32)) > 0)

        def visit(k, depth, found, target):
            pos = pc + sign * k
            if control == "matched":
                inside = (pos >= 0) & (pos < length)
                pos = jnp.clip(pos, 0, length - 1)
            else:
                inside = jnp.ones_like(need)
                pos = (pos + length) % length
            # Two compares on the raw byte instead of a dependent table gather; every
            # opcode table maps each bracket to exactly one byte.
            byte = tape_ref[rows, pos].astype(jnp.int32)
            opening = (byte == open_byte).astype(jnp.int32)
            closing = (byte == close_byte).astype(jnp.int32)
            delta = jnp.where(forward, opening - closing, closing - opening)
            active = need & ~found & inside & (k < length)
            depth = jnp.where(active, depth + delta, depth)
            hit = active & (depth == 0)
            return depth, found | hit, jnp.where(hit, pos, target)

        def body(carry):
            # `search_chunk` positions per warp reduction: the reduction and the loop
            # test cost more than the loads.
            k, depth, found, target = carry
            for j in range(search_chunk):
                depth, found, target = visit(k + j, depth, found, target)
            return k + search_chunk, depth, found, target

        init = (jnp.int32(1), zero + 1, jnp.zeros_like(need), zero - 1)
        _k, _depth, _found, target = jax.lax.while_loop(cond, body, init)
        return target

    def one_step(carry):
        row, pc, head0, head1, direction, live, steps, ops = carry
        # Recomputed from the carried lane ids every step on purpose: as a loop
        # invariant, Triton hoists the row addressing and the step costs 25% more
        # (80% more on jump-heavy soups).
        rows = jnp.where(live, row, dummy)
        cmd = table_ref[tape_ref[rows, pc].astype(jnp.int32)]
        value0 = tape_ref[rows, head0]
        value1 = tape_ref[rows, head1]
        jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
        jump_backward = (cmd == Op.LOOP_END) & (value0 != 0)

        # Writes: at most one byte changes, at head0 or head1; a masked store so
        # nothing is read back or written for the lanes that write nothing.
        write_pos = jnp.where(cmd == Op.COPY01, head1, head0)
        writes = (
            (cmd == Op.PLUS)
            | (cmd == Op.MINUS)
            | (cmd == Op.COPY01)
            | (cmd == Op.COPY10)
        )
        write_val = jnp.where(
            cmd == Op.PLUS,
            value0 + 1,
            jnp.where(
                cmd == Op.MINUS,
                value0 - 1,
                jnp.where(cmd == Op.COPY01, value0, value1),
            ),
        ).astype(jnp.uint8)
        plt.store(tape_ref.at[rows, write_pos], write_val, mask=live & writes)

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
            target = search(rows, pc, jump_forward, live & jumping)
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

        steps = steps + live.astype(jnp.int32)
        ops = ops + (live & is_instruction(cmd)).astype(jnp.int32)
        # A lane stops as soon as it halts or spends its budget; the loop test below
        # runs once per `unroll` steps, so nothing depends on it for exactness.
        finished = new_halted | (steps >= num_steps)
        return (
            row,
            jnp.where(live, new_pc, pc),
            jnp.where(live, new_head0, head0),
            jnp.where(live, new_head1, head1),
            jnp.where(live, new_direction, direction),
            live & ~finished,
            steps,
            ops,
        )

    def loop_cond(carry):
        return jnp.sum(carry[5].astype(jnp.int32)) > 0

    def loop_body(carry):
        for _ in range(unroll):
            carry = one_step(carry)
        return carry

    if heads_from_tape:
        head0 = tape_ref[rows, zero].astype(jnp.int32) % length
        head1 = tape_ref[rows, zero + 1].astype(jnp.int32) % length
        pc = zero + 2
    else:
        pc, head0, head1 = zero, zero, zero
    init = (lanes, pc, head0, head1, zero + 1, valid, zero, zero)
    _r, _pc, _h0, _h1, _d, _live, steps, ops = jax.lax.while_loop(
        loop_cond, loop_body, init
    )
    plt.store(steps_ref.at[rows], steps, mask=valid)
    plt.store(ops_ref.at[rows], ops, mask=valid)


@partial(
    jax.jit,
    static_argnames=(
        "num_steps",
        "heads_from_tape",
        "control",
        "block",
        "search_chunk",
        "unroll",
        "interpret",
    ),
)
def run_kernel(
    tapes: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    heads_from_tape: bool = False,
    control: Control = "matched",
    block: int = 32,
    search_chunk: int = 4,
    unroll: int = 2,
    interpret: bool = False,
) -> tuple[Array, Array, Array]:
    """Run a batch of tapes inside one kernel; same function as `interpreter.run`.

    Args:
        tapes: Unsigned 8-bit array of shape (num_tapes, length).
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values;
            each bracket must map from exactly one byte, as every table built by the
            language module does.
        num_steps: Step budget per tape.
        heads_from_tape: Whether the first two bytes seed the heads.
        control: Control-flow rule; see the interpreter module.
        block: Tapes per kernel block; a block runs until its last tape halts, so
            smaller blocks waste fewer lanes on random soups. One warp is 32.
        search_chunk: Tape positions the bracket search visits between two tests
            of whether any lane is still searching.
        unroll: Steps a block executes between two tests of whether any lane is
            still running.
        interpret: Run the kernel in Pallas interpret mode (any backend, slow).

    Returns:
        A tuple `(tapes, steps, ops)` as `interpreter.run`.

    """
    num, length = tapes.shape
    blocks = -(-num // block)
    padded = blocks * block + 1
    tapes = jnp.concatenate(
        [tapes, jnp.zeros((padded - num, length), dtype=tapes.dtype)]
    )
    kernel = partial(
        _kernel,
        num=num,
        padded=padded,
        heads_from_tape=heads_from_tape,
        num_steps=num_steps,
        length=length,
        block=block,
        control=control,
        search_chunk=search_chunk,
        unroll=unroll,
    )
    tape_spec = pl.BlockSpec((padded, length), lambda i: (0, 0))
    vec_spec = pl.BlockSpec((padded,), lambda i: (0,))
    vec_shape = jax.ShapeDtypeStruct((padded,), jnp.int32)
    table = opcode_table.astype(jnp.int32)
    brackets = jnp.stack(
        [jnp.argmax(table == Op.LOOP_START), jnp.argmax(table == Op.LOOP_END)]
    ).astype(jnp.int32)
    out_tapes, steps, ops = pl.pallas_call(
        kernel,
        grid=(blocks,),
        in_specs=[
            pl.BlockSpec((256,), lambda i: (0,)),
            pl.BlockSpec((2,), lambda i: (0,)),
            tape_spec,
        ],
        out_specs=(tape_spec, vec_spec, vec_spec),
        out_shape=(
            jax.ShapeDtypeStruct((padded, length), tapes.dtype),
            vec_shape,
            vec_shape,
        ),
        input_output_aliases={2: 0},
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=max(1, block // 32)),
    )(table, brackets, tapes)
    return out_tapes[:num], steps[:num], ops[:num]


__all__ = ["run_kernel"]
