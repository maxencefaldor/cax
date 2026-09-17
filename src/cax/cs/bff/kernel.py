"""One-kernel BFF interpreter for GPUs.

`run_kernel` computes the same function as `interpreter.run` inside a single Pallas
kernel: a warp of tapes loops over the step budget on-chip and returns as soon as its
last tape has halted, where the XLA scan pays tens of kernel launches per step for
every tape. The bracket search is the reference's walk, vectorised over the warp and
stopped once every searching lane has its answer; each lane caches its last forward
and backward result until a write lands in the walked range, so a loop searches once.

The kernel targets the Triton backend and runs under `interpret=True` elsewhere for
tests. Tapes and the per-tape state are updated in place through input-output aliases.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .interpreter import Control
from .language import Op, is_instruction

# Rows of the per-tape state array.
PC, HEAD0, HEAD1, DIRECTION, HALTED, STEPS, OPS, FIRST = range(8)


def _kernel(
    table_ref,
    brackets_ref,
    tape_in_ref,
    state_in_ref,
    tape_ref,
    state_ref,
    match_ref,
    *,
    num: int,
    padded: int,
    num_steps: int,
    length: int,
    block: int,
    control: Control,
    search_chunk: int,
    unroll: int,
    fresh: bool,
) -> None:
    del tape_in_ref, state_in_ref  # aliased to tape_ref and state_ref
    # Dead lanes point at a dummy row so their masked accesses never alias a live one.
    lanes = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = lanes < num
    dummy = padded - 1
    zero = jnp.zeros((block,), dtype=jnp.int32)
    false = jnp.zeros((block,), dtype=bool)
    open_byte = brackets_ref[0]
    close_byte = brackets_ref[1]

    def search(rows: Array, pc: Array, forward: Array, need: Array) -> Array:
        """Position of the matching bracket for the lanes in `need`, else -1."""
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
            byte = tape_ref[rows, pos].astype(jnp.int32)
            opening = (byte == open_byte).astype(jnp.int32)
            closing = (byte == close_byte).astype(jnp.int32)
            delta = jnp.where(forward, opening - closing, closing - opening)
            active = need & ~found & inside & (k < length)
            depth = jnp.where(active, depth + delta, depth)
            hit = active & (depth == 0)
            return depth, found | hit, jnp.where(hit, pos, target)

        def body(carry):
            k, depth, found, target = carry
            for j in range(search_chunk):
                depth, found, target = visit(k + j, depth, found, target)
            return k + search_chunk, depth, found, target

        init = (jnp.int32(1), zero + 1, false, zero - 1)
        return jax.lax.while_loop(cond, body, init)[3]

    def visited(start: Array, target: Array, pos: Array, forward: bool) -> Array:
        """Whether a write at `pos` lands where the walk from `start` looked."""
        if control == "matched":
            lo, hi = (start + 1, target) if forward else (target, start - 1)
            within = (pos >= lo) & (pos <= hi)
            everything = pos > start if forward else pos < start
        else:
            dist = (pos - start if forward else start - pos) % length
            span = (target - start if forward else start - target) % length
            within = (dist >= 1) & (dist <= span)
            everything = pos != start
        return jnp.where(target < 0, everything, within) | (pos == start)

    def one_step(carry):
        row, pc, head0, head1, direction, live, steps, ops, first, cache, census = carry
        (fpc, ftgt, fok), (bpc, btgt, bok), bits = cache
        n_open, n_close = census
        # Recomputed every step on purpose: hoisted, the addressing costs 25% more.
        rows = jnp.where(live, row, dummy)
        cmd = table_ref[tape_ref[rows, pc].astype(jnp.int32)]
        value0 = tape_ref[rows, head0]
        value1 = tape_ref[rows, head1]
        jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
        jump_backward = (cmd == Op.LOOP_END) & (value0 != 0)

        # Writes: at most one byte, at head0 or head1.
        write_pos = jnp.where(cmd == Op.COPY01, head1, head0)
        writing = live & (
            (cmd == Op.PLUS)
            | (cmd == Op.MINUS)
            | (cmd == Op.COPY01)
            | (cmd == Op.COPY10)
        )
        write_val = jnp.where(
            cmd == Op.PLUS,
            value0 + 1,
            jnp.where(
                cmd == Op.MINUS, value0 - 1, jnp.where(cmd == Op.COPY01, value0, value1)
            ),
        ).astype(jnp.uint8)
        # The overwritten byte is already loaded: value1 under `.`, else value0.
        old = jnp.where(cmd == Op.COPY01, value1, value0).astype(jnp.int32)
        new = write_val.astype(jnp.int32)
        # Only a write that makes or unmakes a bracket can change a search.
        was = (old == open_byte) | (old == close_byte)
        becomes = (new == open_byte) | (new == close_byte)
        rewrite = writing & (was | becomes)
        if control == "matched":
            fok = fok & ~(rewrite & visited(fpc, ftgt, write_pos, True))
            bok = bok & ~(rewrite & visited(bpc, btgt, write_pos, False))
        elif control == "cyclic":
            bits = tuple(jnp.where(rewrite, 0, b) for b in bits)
        if control == "cyclic":
            n_open += jnp.where(writing, (new == open_byte) * 1 - (old == open_byte), 0)
            n_close += jnp.where(
                writing, (new == close_byte) * 1 - (old == close_byte), 0
            )
        plt.store(tape_ref.at[rows, write_pos], write_val, mask=writing)

        # Head moves, wrapping on the tape; `~` exchanges the heads.
        inc0 = (cmd == Op.INC0).astype(jnp.int32) - (cmd == Op.DEC0).astype(jnp.int32)
        inc1 = (cmd == Op.INC1).astype(jnp.int32) - (cmd == Op.DEC1).astype(jnp.int32)
        swap = cmd == Op.SWAP
        new_head0 = jnp.where(swap, head1, (head0 + inc0 + length) % length)
        new_head1 = jnp.where(swap, head0, (head1 + inc1 + length) % length)

        # Control flow.
        jumping = jump_forward | jump_backward
        if control == "flip":
            new_direction = jnp.where(jumping, -direction, direction)
            new_pc = (pc + new_direction + length) % length
            new_halted = false
        else:
            new_direction = direction
            word, bit = pc >> 5, 1 << (pc & 31)  # the table's slot for this position
            if control == "matched":
                # Two entries, the last forward and the last backward search.
                hit_f = jump_forward & fok & (fpc == pc)
                hit_b = jump_backward & bok & (bpc == pc)
                hit = hit_f | hit_b
                known = jnp.where(hit_f, ftgt, btgt)
                need = live & jumping & ~hit
            else:
                # One entry per position, in a scratch table; a bit per position says
                # whether it is current. On a ring an unmatched bracket walks the whole
                # tape, so the census also skips walks with no partner byte at all.
                mask = bits[0]
                for w in range(1, length // 32):
                    mask = jnp.where(word == w, bits[w], mask)
                hit = jumping & ((mask & bit) != 0)
                known = match_ref[rows, pc]
                need = live & jumping & ~hit
                need = need & jnp.where(jump_forward, n_close > 0, n_open > 0)
            walked = search(rows, pc, jump_forward, need)
            target = jnp.where(hit, known, walked)
            if control == "matched":
                fresh_f = need & jump_forward
                fresh_b = need & jump_backward
                fpc, ftgt, fok = (
                    jnp.where(fresh_f, pc, fpc),
                    jnp.where(fresh_f, target, ftgt),
                    fok | fresh_f,
                )
                bpc, btgt, bok = (
                    jnp.where(fresh_b, pc, bpc),
                    jnp.where(fresh_b, target, btgt),
                    bok | fresh_b,
                )
            else:
                plt.store(match_ref.at[rows, pc], target, mask=need)
                bits = tuple(
                    jnp.where(need & (word == w), b | bit, b)
                    for w, b in enumerate(bits)
                )
            found = target >= 0
            jumped = jnp.where(found, target, pc)
            if control == "matched":  # an unmatched taken jump lands off the tape
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
                new_halted = false

        steps = steps + live.astype(jnp.int32)
        ops = ops + (live & is_instruction(cmd)).astype(jnp.int32)
        first = first + (live & (pc < length // 2)).astype(jnp.int32)
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
            first,
            ((fpc, ftgt, fok), (bpc, btgt, bok), bits),
            (n_open, n_close),
        )

    def loop_cond(carry):
        return jnp.sum(carry[5].astype(jnp.int32)) > 0  # any lane live

    def loop_body(carry):
        for _ in range(unroll):
            carry = one_step(carry)
        return carry

    rows = jnp.where(valid, lanes, dummy)
    if fresh:  # a run from the start: constants, not loads
        pc, head0, head1, steps, ops, first = (zero,) * 6
        halted = false
    else:
        pc, head0, head1, _d, halted, steps, ops, first = (
            state_ref[i, rows] for i in range(8)
        )
        halted = halted != 0
    direction = state_ref[DIRECTION, rows] if control == "flip" else zero + 1
    live = valid & ~halted & (steps < num_steps)
    # Both caches ride in the carry; the unused one is constant and costs nothing.
    entries = ((zero, zero, false), (zero, zero, false))
    cache = (*entries, tuple(zero for _ in range(length // 32)))
    census = (zero, zero)
    if control == "cyclic":  # bracket bytes on each tape, kept exact under writes

        def count(k: int, c: tuple[Array, Array]) -> tuple[Array, Array]:
            byte = tape_ref[rows, zero + k].astype(jnp.int32)
            return c[0] + (byte == open_byte), c[1] + (byte == close_byte)

        census = jax.lax.fori_loop(0, length, count, census)
    init = (lanes, pc, head0, head1, direction, live, steps, ops, first, cache, census)
    out = jax.lax.while_loop(loop_cond, loop_body, init)
    pc, head0, head1, direction, live, steps, ops, first = out[1:9]
    halted = ~live & (steps < num_steps)  # stopped short of the budget
    final = (pc, head0, head1, direction, halted.astype(jnp.int32), steps, ops, first)
    for i, value in enumerate(final):
        plt.store(state_ref.at[i, rows], value, mask=valid)


def _launch(
    tapes: Array,
    state: Array,
    opcode_table: Array,
    *,
    num_steps: int,
    control: Control,
    block: int,
    search_chunk: int,
    unroll: int,
    interpret: bool,
    fresh: bool = False,
) -> tuple[Array, Array]:
    """Run every tape from `state` until it halts or has spent `num_steps` in total."""
    num, length = tapes.shape
    blocks = -(-num // block)
    padded = blocks * block + 1  # spare lanes of the last block, plus the dummy row
    pad = padded - num
    tapes = jnp.concatenate([tapes, jnp.zeros((pad, length), tapes.dtype)])
    state = jnp.concatenate([state, jnp.zeros((8, pad), jnp.int32)], axis=1)
    table = opcode_table.astype(jnp.int32)
    brackets = jnp.stack(
        [jnp.argmax(table == Op.LOOP_START), jnp.argmax(table == Op.LOOP_END)]
    ).astype(jnp.int32)
    kernel = partial(
        _kernel,
        num=num,
        padded=padded,
        num_steps=num_steps,
        length=length,
        block=block,
        control=control,
        search_chunk=search_chunk,
        unroll=unroll,
        fresh=fresh,
    )
    tape_spec = pl.BlockSpec((padded, length), lambda i: (0, 0))
    state_spec = pl.BlockSpec((8, padded), lambda i: (0, 0))
    tapes, state, _ = pl.pallas_call(
        kernel,
        grid=(blocks,),
        in_specs=[
            pl.BlockSpec((256,), lambda i: (0,)),
            pl.BlockSpec((2,), lambda i: (0,)),
            tape_spec,
            state_spec,
        ],
        out_specs=(tape_spec, state_spec, tape_spec),
        out_shape=(
            jax.ShapeDtypeStruct((padded, length), tapes.dtype),
            jax.ShapeDtypeStruct((8, padded), jnp.int32),
            jax.ShapeDtypeStruct((padded, length), jnp.int32),  # cyclic match table
        ),
        input_output_aliases={2: 0, 3: 1},
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=max(1, block // 32)),
    )(table, brackets, tapes, state)
    return tapes[:num], state[:, :num]


@partial(
    jax.jit,
    static_argnames=(
        "num_steps",
        "heads_from_tape",
        "control",
        "block",
        "search_chunk",
        "unroll",
        "first",
        "capacity",
        "two_phase_min",
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
    search_chunk: int | None = None,
    unroll: int = 2,
    first: int = 256,
    capacity: float = 1 / 8,
    two_phase_min: int = 1 << 18,
    interpret: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Run a batch of tapes inside one kernel; same function as `interpreter.run`.

    With `"matched"` control and at least `two_phase_min` tapes the run has two
    phases: every tape for `first` steps, then only the tapes still running, gathered
    into a buffer of `capacity` times the batch, for the rest of the budget (all of
    them if more survive than fit). A warp lives as long as its longest tape, so this
    keeps the warps of halted tapes from idling through the budget; it pays once the
    survivors are enough warps to fill the GPU. The result is the same either way.

    Args:
        tapes: Unsigned 8-bit array of shape (num_tapes, length).
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values;
            each bracket must map from exactly one byte, as every table built by the
            language module does.
        num_steps: Step budget per tape.
        heads_from_tape: Whether the first two bytes seed the heads.
        control: Control-flow rule; see the interpreter module.
        block: Tapes per kernel block; one warp is 32.
        search_chunk: Tape positions the bracket search visits between two tests of
            whether any lane is still searching; 4 by default, 16 for `"cyclic"`.
        unroll: Steps a block executes between two tests of whether any lane runs.
        first: Steps of the first phase.
        capacity: Survivor buffer size as a fraction of the batch.
        two_phase_min: Smallest batch that runs in two phases.
        interpret: Run the kernel in Pallas interpret mode (any backend, slow).

    Returns:
        A tuple `(tapes, steps, ops)` as `interpreter.run`.

    """
    num, length = tapes.shape
    if search_chunk is None:
        search_chunk = 16 if control == "cyclic" else 4
    launch = partial(
        _launch,
        num_steps=num_steps,
        control=control,
        block=block,
        search_chunk=search_chunk,
        unroll=unroll,
        interpret=interpret,
    )
    zero = jnp.zeros((num,), jnp.int32)
    if heads_from_tape:
        pc = zero + 2
        head0 = tapes[:, 0].astype(jnp.int32) % length
        head1 = tapes[:, 1].astype(jnp.int32) % length
    else:
        pc, head0, head1 = zero, zero, zero
    state = jnp.stack([pc, head0, head1, zero + 1, zero, zero, zero, zero])

    fresh = not heads_from_tape
    if control != "matched" or first >= num_steps or num < two_phase_min:
        tapes, state = launch(tapes, state, opcode_table, fresh=fresh)
        return tapes, state[STEPS], state[OPS], state[FIRST]

    tapes, state = launch(tapes, state, opcode_table, num_steps=first, fresh=fresh)
    alive = state[HALTED] == 0
    count = jnp.sum(alive)
    size = max(block, round(capacity * num))

    def survivors(args: tuple[Array, Array]) -> tuple[Array, Array]:
        tapes, state = args
        (chosen,) = jnp.nonzero(alive, size=size, fill_value=0)
        used = jnp.arange(size) < count
        sub_state = state[:, chosen].at[HALTED].set(jnp.where(used, 0, 1))
        sub_tapes, sub_state = launch(tapes[chosen], sub_state, opcode_table)
        rows = jnp.where(used, chosen, num)  # out of bounds: dropped
        return (
            tapes.at[rows].set(sub_tapes, mode="drop"),
            state.at[:, rows].set(sub_state, mode="drop"),
        )

    def everyone(args: tuple[Array, Array]) -> tuple[Array, Array]:
        return launch(*args, opcode_table)

    tapes, state = jax.lax.cond(count <= size, survivors, everyone, (tapes, state))
    return tapes, state[STEPS], state[OPS], state[FIRST]


__all__ = ["run_kernel"]
