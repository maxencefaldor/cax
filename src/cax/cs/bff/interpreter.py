"""BFF interpreter module.

This module implements the BFF virtual machine as a pure, vectorised function over a
batch of tapes. A tape is both the program and its memory: the instruction pointer and
the two data heads all address the same byte array, so a program can rewrite itself and
its neighbour while running. With the default control flow the semantics follow the
reference implementation byte for byte; see `step` for the rules that matter (loop
tests, unmatched brackets, halting).

Control flow is a parameter, because it is the only rule of BFF that is not local (a
bracket's meaning depends on a partner somewhere else on the tape) and not total (a
missing partner halts the thread):

- `"matched"`: the reference. `[` and `]` jump to their matching partner, with
  nesting; a taken jump with no partner halts, and so does running off the end.
- `"cyclic"`: the tape is a ring. The partner is searched cyclically; a bracket with
  no partner is a no-op. Nothing halts but the step budget.
- `"flip"`: the tape is a ring and the thread carries a direction. `]` reverses it
  when the byte under head0 is nonzero, `[` when it is zero, and no search is ever
  made: a bracket is a mirror that reflects the instruction pointer from either side.
  Two mirrors of the same kind enclose a room: a pointer inside `] body ]` bounces
  through the body forward and backward while the test byte is nonzero and leaves
  through whichever wall opens when it reaches zero; a pointer outside can only enter
  while the byte is zero. Nesting has no meaning. O(1) per step.

Two implementations of one step live here. `step` is the readable one: one thread, one
instruction, written the way the rules are stated. `run` is the fast one: it executes a
whole batch and only performs the O(length) bracket search for the threads that take a
jump on that step, gathered into a fixed-capacity buffer, with a full search as the
fallback when the buffer overflows. Both compute the same function; the tests check it.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108
    [2] cubff reference implementation, `bff.inc.h`.
        https://github.com/paradigms-of-intelligence/cubff

"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .language import Op, is_instruction

Control = Literal["matched", "cyclic", "flip"]
Implementation = Literal["auto", "xla", "kernel"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ThreadState:
    """Execution state of one tape between two steps.

    Attributes:
        tape: Unsigned 8-bit array of shape (length,), the program and its memory.
        pc: Scalar int32, the instruction pointer.
        head0: Scalar int32 in [0, length), the read head.
        head1: Scalar int32 in [0, length), the write head.
        direction: Scalar int32, +1 or -1, the way the instruction pointer moves. Only
            the `"flip"` control flow ever sets it to -1.
        halted: Scalar bool; once set, the thread is a no-op for the remaining steps.
        steps: Scalar int32, number of steps executed so far, no-ops included.
        ops: Scalar int32, number of executed steps whose byte was an instruction.
        first: Scalar int32, number of executed steps with the pointer in the first
            half of the tape; the rest were spent in the second half.

    """

    tape: Array
    pc: Array
    head0: Array
    head1: Array
    direction: Array
    halted: Array
    steps: Array
    ops: Array
    first: Array


def initial_thread_state(tape: Array, *, heads_from_tape: bool) -> ThreadState:
    """Create the thread state at the start of an execution.

    Two conventions exist in the reference implementation. With `heads_from_tape` false
    (`bff_noheads`, the language of Section 2 of the paper) the instruction pointer and
    both heads start at zero. With it true (`bff`) the heads start at the values of the
    first two bytes modulo the tape length and the instruction pointer starts at 2.

    Args:
        tape: Unsigned 8-bit array of shape (length,).
        heads_from_tape: Whether the first two bytes seed the head positions.

    Returns:
        A fresh `ThreadState`.

    """
    length = tape.shape[-1]
    zero = jnp.zeros((), dtype=jnp.int32)
    if heads_from_tape:
        head0 = tape[0].astype(jnp.int32) % length
        head1 = tape[1].astype(jnp.int32) % length
        pc = jnp.full((), 2, dtype=jnp.int32)
    else:
        head0 = zero
        head1 = zero
        pc = zero
    return ThreadState(
        tape=tape,
        pc=pc,
        head0=head0,
        head1=head1,
        direction=jnp.ones((), dtype=jnp.int32),
        halted=jnp.zeros((), dtype=bool),
        steps=zero,
        ops=zero,
        first=zero,
    )


def match_bracket(tape: Array, pc: Array, forward: Array, opcode_table: Array) -> Array:
    """Find the bracket matching the one under the instruction pointer.

    Works from the inclusive prefix sum of bracket depth over the tape. The `]` matching
    a `[` at `pc` is the first position after it whose depth after equals the depth
    before the `[`; the `[` matching a `]` at `pc` is the last position before it whose
    depth before is one less than the depth before the `]`. Both are the positions
    where the reference implementation's counting scan reaches zero.

    Args:
        tape: Unsigned 8-bit array of shape (..., length).
        pc: Integer array of shape (...), position of the bracket.
        forward: Boolean array of shape (...); True to match a `[`, False a `]`.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.

    Returns:
        Integer array of shape (...) with the matching position, or -1 if none.

    """
    length = tape.shape[-1]
    ops = opcode_table[tape]
    delta = (ops == Op.LOOP_START).astype(jnp.int16) - (ops == Op.LOOP_END).astype(
        jnp.int16
    )
    depth_after = jnp.cumsum(delta, axis=-1, dtype=jnp.int16)
    depth_before = depth_after - delta
    depth = jnp.where(forward[..., None], depth_after, depth_before)
    at_pc = jnp.take_along_axis(depth, pc[..., None], axis=-1)
    position = jnp.arange(length, dtype=jnp.int32)
    side = jnp.where(
        forward[..., None], position > pc[..., None], position < pc[..., None]
    )
    candidate = side & (depth == at_pc - 1)
    # The first candidate forward, the last one backward: one max of a signed position.
    score = jnp.where(
        candidate, jnp.where(forward[..., None], -position, position), -length - 1
    )
    best = jnp.max(score, axis=-1)
    return jnp.where(best < -length, -1, jnp.abs(best))


def match_bracket_cyclic(
    tape: Array, pc: Array, forward: Array, opcode_table: Array
) -> Array:
    """Find the matching bracket on a ring, scanning away from `pc` in one direction.

    The scan walks `pc + 1, pc + 2, ...` (or `pc - 1, ...`) modulo the length, counting
    brackets of the scanned kind as -1 and of the other kind as +1 from a running total
    of 1, and stops where the total reaches zero, exactly the reference's scan but
    wrapping. If it returns to `pc` without reaching zero there is no partner.

    Args:
        tape: Unsigned 8-bit array of shape (..., length).
        pc: Integer array of shape (...), position of the bracket.
        forward: Boolean array of shape (...); True to match a `[`, False a `]`.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.

    Returns:
        Integer array of shape (...) with the matching position, or -1 if none.

    """
    length = tape.shape[-1]
    offset = jnp.arange(length, dtype=jnp.int32)
    sign = jnp.where(forward, 1, -1)[..., None]
    walked = (pc[..., None] + sign * offset) % length
    ops = opcode_table[jnp.take_along_axis(tape, walked, axis=-1)]
    opening = (ops == Op.LOOP_START).astype(jnp.int16)
    closing = (ops == Op.LOOP_END).astype(jnp.int16)
    delta = jnp.where(forward[..., None], opening - closing, closing - opening)
    delta = delta.at[..., 0].set(1)  # the bracket at pc itself opens the count
    total = jnp.cumsum(delta, axis=-1, dtype=jnp.int16)
    candidate = (offset > 0) & (total == 0)
    first = jnp.argmax(candidate, axis=-1)
    found = jnp.any(candidate, axis=-1)
    target = jnp.take_along_axis(walked, first[..., None], axis=-1)[..., 0]
    return jnp.where(found, target, -1)


def _execute(
    state: ThreadState,
    cmd: Array,
    value0: Array,
    value1: Array,
    target: Array,
    control: Control,
) -> ThreadState:
    """Apply one decoded instruction; shared by `step` and the batched fast path.

    `target` is the matching bracket position for this thread (or -1), already
    computed for the `"matched"` and `"cyclic"` control flows; unused for `"flip"`.
    Arrays may carry a leading batch axis.
    """
    tape = state.tape
    length = tape.shape[-1]
    pc, head0, head1 = state.pc, state.head0, state.head1
    batch = tape.shape[:-1]
    index = tuple(jnp.indices(batch)) if batch else ()

    # Writes: at most one byte changes, at head0 or head1.
    write_pos = jnp.where(cmd == Op.COPY01, head1, head0)
    write_val = jnp.select(
        [cmd == Op.PLUS, cmd == Op.MINUS, cmd == Op.COPY01, cmd == Op.COPY10],
        [value0 + 1, value0 - 1, value0, value1],
        default=tape[(*index, write_pos)],
    ).astype(jnp.uint8)
    new_tape = tape.at[(*index, write_pos)].set(write_val)

    # Head moves, wrapping on the tape; `~` exchanges the heads.
    moved0 = (head0 + (cmd == Op.INC0) - (cmd == Op.DEC0)) % length
    moved1 = (head1 + (cmd == Op.INC1) - (cmd == Op.DEC1)) % length
    swap = cmd == Op.SWAP
    new_head0 = jnp.where(swap, head1, moved0)
    new_head1 = jnp.where(swap, head0, moved1)

    # Control flow.
    jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
    jump_backward = (cmd == Op.LOOP_END) & (value0 != 0)
    direction = state.direction
    if control == "flip":
        direction = jnp.where(jump_forward | jump_backward, -direction, direction)
        new_pc = (pc + direction) % length
        new_halted = jnp.zeros_like(state.halted)
    else:
        found = target >= 0
        jumped = jnp.where(found, target, pc)
        if control == "matched":
            # An unmatched taken jump lands off the tape: halt after the increment.
            jumped = jnp.where(
                jump_forward & ~found,
                length,
                jnp.where(jump_backward & ~found, -2, jumped),
            )
        new_pc = jnp.where(jump_forward | jump_backward, jumped, pc) + 1
        if control == "matched":
            new_halted = (new_pc < 0) | (new_pc >= length)
            new_pc = jnp.clip(new_pc, 0, length - 1)
        else:
            new_pc = new_pc % length
            new_halted = jnp.zeros_like(state.halted)

    live = ~state.halted

    def keep(new: Array, old: Array) -> Array:
        return jnp.where(live, new, old)

    return ThreadState(
        tape=jnp.where(live[..., None], new_tape, tape),
        pc=keep(new_pc, pc),
        head0=keep(new_head0, head0),
        head1=keep(new_head1, head1),
        direction=keep(direction, state.direction),
        halted=state.halted | new_halted,
        steps=state.steps + live,
        ops=state.ops + (live & is_instruction(cmd)),
        first=state.first + (live & (pc < length // 2)),
    )


def step(
    state: ThreadState, opcode_table: Array, *, control: Control = "matched"
) -> ThreadState:
    """Execute one instruction of one thread.

    The rules, as in the reference implementation:

    - The byte under the instruction pointer is decoded with `opcode_table`. Bytes that
      are not instructions (`NOOP`, `NULL`) do nothing but still consume a step.
    - `+ - . ,` write one byte at a head. `< > { }` move a head by one, wrapping. `~`
      (only in the swap-heads dialect) exchanges the heads.
    - `[` tests the byte under head0: if it is zero, the instruction pointer jumps to
      the matching `]`, with nesting. `]` jumps back to the matching `[` if the byte
      is nonzero. Any nonzero byte counts as true, instruction bytes included.
    - A taken jump with no matching bracket halts the thread. So does the instruction
      pointer leaving the tape. Untaken jumps never scan and never halt.
    - After the instruction the pointer advances by one, from the matched bracket when a
      jump was taken.

    The `"cyclic"` and `"flip"` control flows change the bracket rules as described in
    the module docstring and never halt.

    Args:
        state: Thread state before the step.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        control: Control-flow rule.

    Returns:
        Thread state after the step. Halted threads are returned unchanged.

    """
    tape = state.tape
    cmd = opcode_table[tape[state.pc]]
    value0 = tape[state.head0]
    value1 = tape[state.head1]
    jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
    if control == "matched":
        target = match_bracket(tape, state.pc, jump_forward, opcode_table)
    elif control == "cyclic":
        target = match_bracket_cyclic(tape, state.pc, jump_forward, opcode_table)
    else:
        target = jnp.full((), -1, dtype=jnp.int32)
    return _execute(state, cmd, value0, value1, target, control)


def _step_batch(
    state: ThreadState, opcode_table: Array, scan_capacity: int, control: Control
) -> ThreadState:
    """One step of a batch of threads; same function as `jax.vmap(step)`, faster.

    Everything but the bracket search is O(1) per thread. The search is done only for
    the threads that take a jump this step: their indices are gathered into a buffer of
    `scan_capacity` slots; if more threads need it than fit, every thread is searched.
    Either way the result is identical.
    """
    tape = state.tape
    num = tape.shape[0]
    index = jnp.arange(num)
    pc = state.pc

    cmd = opcode_table[tape[index, pc]]
    value0 = tape[index, state.head0]
    value1 = tape[index, state.head1]
    jump_forward = (cmd == Op.LOOP_START) & (value0 == 0)
    jump_backward = (cmd == Op.LOOP_END) & (value0 != 0)

    if control == "flip":
        target = jnp.full((num,), -1, dtype=jnp.int32)
    else:
        search = match_bracket if control == "matched" else match_bracket_cyclic
        need = ~state.halted & (jump_forward | jump_backward)
        count = jnp.sum(need)

        def search_some() -> Array:
            (chosen,) = jnp.nonzero(need, size=scan_capacity, fill_value=0)
            found = search(tape[chosen], pc[chosen], jump_forward[chosen], opcode_table)
            # Unused slots all point at thread 0 and all carry thread 0's answer, so
            # the scatter is deterministic; threads that did not jump ignore it.
            return jnp.full((num,), -1, dtype=jnp.int32).at[chosen].set(found)

        def search_all() -> Array:
            return search(tape, pc, jump_forward, opcode_table)

        target = jax.lax.cond(count <= scan_capacity, search_some, search_all)

    return _execute(state, cmd, value0, value1, target, control)


def _run_batch(
    state: ThreadState,
    opcode_table: Array,
    num_steps: int,
    scan_capacity: int,
    control: Control,
) -> ThreadState:
    body = partial(
        _step_batch,
        opcode_table=opcode_table,
        scan_capacity=scan_capacity,
        control=control,
    )
    return jax.lax.fori_loop(0, num_steps, lambda _, s: body(s), state)


@partial(
    jax.jit,
    static_argnames=(
        "num_steps",
        "heads_from_tape",
        "control",
        "scan_fraction",
        "compact_after",
        "compact_fraction",
        "implementation",
    ),
)
def run(
    tapes: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    heads_from_tape: bool = False,
    control: Control = "matched",
    scan_fraction: float = 1 / 16,
    compact_after: int = 256,
    compact_fraction: float = 1 / 8,
    implementation: Implementation = "auto",
) -> tuple[Array, Array, Array, Array]:
    """Run a batch of tapes, each for at most `num_steps` steps.

    Every tape is executed independently for the same result as `step` applied
    `num_steps` times. On a GPU the work is done by `kernel.run_kernel`, one Pallas
    kernel that loops over the steps on-chip and exits a block as soon as its tapes
    have all halted; everywhere else by the XLA scan below, where two devices keep
    the fixed-shape scan from paying for work that most tapes do not need, without
    changing the result:

    - Bracket searches run only for the tapes taking a jump on a step, gathered into a
      buffer of `scan_fraction` times the batch; a step where more tapes jump than fit
      searches all of them.
    - After `compact_after` steps the tapes still running are gathered into a buffer of
      `compact_fraction` times the batch and only they run the remaining steps; if more
      are alive than fit, all tapes run them.

    On a uniformly random soup about 8% of pairs survive 256 steps and fewer than 1%
    jump on any given step; the defaults follow. Soups full of replicators keep more
    tapes alive and jumping, so their epochs cost more; that is the physics, not the
    implementation. The `"cyclic"` and `"flip"` control flows never halt, so every
    tape runs the whole budget and compaction is skipped.

    Args:
        tapes: Unsigned 8-bit array of shape (num_tapes, length).
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per tape; the reference uses 8192.
        heads_from_tape: Whether the first two bytes seed the heads; see
            `initial_thread_state`.
        control: Control-flow rule; see the module docstring.
        scan_fraction: Bracket-search buffer size as a fraction of the batch.
        compact_after: Steps run on the whole batch before compacting the survivors;
            at least `num_steps` disables compaction.
        compact_fraction: Survivor buffer size as a fraction of the batch.
        implementation: `"kernel"` for the Pallas kernel, `"xla"` for the scan, or
            `"auto"` to pick the kernel on GPU and the scan elsewhere.

    Returns:
        A tuple `(tapes, steps, ops, first)`: the tapes after execution, with the same
            shape as the input; the number of steps each tape executed before halting
            or exhausting the budget; the number of those steps that executed an
            instruction rather than a data byte; and the number spent with the pointer
            in the first half of the tape.

    """
    if implementation == "auto":
        implementation = "kernel" if jax.default_backend() == "gpu" else "xla"
    if implementation == "kernel":
        from .kernel import run_kernel

        return run_kernel(
            tapes,
            opcode_table,
            num_steps=num_steps,
            heads_from_tape=heads_from_tape,
            control=control,
        )
    num = tapes.shape[0]
    init = jax.vmap(partial(initial_thread_state, heads_from_tape=heads_from_tape))
    state = init(tapes)

    def capacity(fraction: float, size: int) -> int:
        return max(1, min(size, round(fraction * size)))

    scan_capacity = capacity(scan_fraction, num)
    first = min(compact_after, num_steps) if control == "matched" else num_steps
    state = _run_batch(state, opcode_table, first, scan_capacity, control)
    remaining = num_steps - first
    if remaining == 0:
        return state.tape, state.steps, state.ops, state.first

    compact_capacity = capacity(compact_fraction, num)
    alive = ~state.halted
    count = jnp.sum(alive)

    def run_compacted(state: ThreadState) -> ThreadState:
        (chosen,) = jnp.nonzero(alive, size=compact_capacity, fill_value=0)
        subset = jax.tree.map(lambda leaf: leaf[chosen], state)
        # Unused slots alias thread 0 (consistent values, deterministic scatter); a
        # halted thread 0 is a no-op either way.
        subset = _run_batch(
            subset,
            opcode_table,
            remaining,
            capacity(scan_fraction, compact_capacity),
            control,
        )
        return jax.tree.map(lambda full, part: full.at[chosen].set(part), state, subset)

    def run_all(state: ThreadState) -> ThreadState:
        return _run_batch(state, opcode_table, remaining, scan_capacity, control)

    state = jax.lax.cond(count <= compact_capacity, run_compacted, run_all, state)
    return state.tape, state.steps, state.ops, state.first


__all__ = [
    "Control",
    "Implementation",
    "ThreadState",
    "initial_thread_state",
    "match_bracket",
    "match_bracket_cyclic",
    "run",
    "step",
]
