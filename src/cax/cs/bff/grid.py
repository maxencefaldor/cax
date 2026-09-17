"""BFF grid module.

This module implements a spatial substrate for BFF: memory is one byte array on a torus
of any dimension, and threads walk it. A thread is a position, a direction and two head
positions; each step it reads the byte under its pointer, executes it, and moves one
cell in its direction. There are no tapes and no pairing: threads interact because they
share the memory. The state is the picture.

The instruction set is BFF's. The brackets turn instead of jumping: `[` turns left when
the byte under head0 is zero, `]` turns right or reverses (see `Turn`) when it is
nonzero. A direction is an index in the ring of the `2d` signed axes, `+x, +y, ...,
-x, -y, ...`, and a turn is a step in that ring, so in one dimension every turn is a
reversal and the machine is the `flip` control flow of `interpreter`. Heads move along
the axis the thread is travelling on. Every instruction is total; a thread ends only
with its step budget, then respawns elsewhere.

See `notes/09_grid_design.md` for the reasons behind each choice.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108

"""

import math
from dataclasses import dataclass
from typing import Literal, override

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem

from .language import Op, opcode_table_from_string

Control = Literal["flip", "cyclic"]
"""How brackets work.

`"flip"`: brackets turn, see `Turn`. `"cyclic"`: brackets jump as in the `cyclic`
control flow of `interpreter`, along the line the thread travels: `[` with a zero test
byte jumps ahead to its matching `]`, `]` with a nonzero one jumps back to its matching
`[`, both with nesting and wrapping around the torus, and a bracket with no partner on
its line is a no-op. Threads never turn, so in two dimensions the machine is a weave of
one-dimensional machines on rows and columns that share every cell. The search costs
one line of memory per thread per step.
"""

Headings = Literal["both", "positive"]
"""Which directions threads spawn with: any of the `2d` signed axes, or only the `d`
positive ones. Executing a stretch of memory backwards is not executing its mirror
image, because heads move with absolute sign, so a replicator laid along a line is
copied by forward threads and damaged by backward ones; `"positive"` removes the
damage. Threads still reverse under `"flip"` brackets.
"""

Heads = Literal["absolute", "relative"]
"""How head moves read the thread's heading. `"absolute"`: `>` moves head0 towards
`+axis` whichever way the thread travels, as in the 1D interpreter. `"relative"`: `>`
moves head0 the way the thread travels, so a thread running a stretch backwards
executes its mirror image exactly, and a replicator on a line is copied by threads
heading either way. Not for `"flip"` loops, whose backward pass would then undo the
forward pass's head moves.
"""

Turn = Literal["quarter", "reflect"]
"""What `]` does when it turns: a quarter turn like `[`, or a reversal.

In one dimension both are the reversal of the `flip` machine. In two or more, a
quarter turn makes every loop a closed path with `2d` corners, which random memory
almost never contains, while a reflecting `]` keeps the one-dimensional bounce loop
`] body ]` available along any line and leaves `[` to change axis.
"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GridState:
    """State of the grid machine.

    Attributes:
        memory: Unsigned 8-bit array of shape `dims`, the shared code and data.
        position: Int32 array of shape (num_threads, len(dims)), the instruction
            pointers.
        direction: Int32 array of shape (num_threads,), each in `[0, 2 * len(dims))`:
            `k` is `+axis k` for `k < len(dims)` and `-axis (k - len(dims))` otherwise.
        head0: Int32 array of shape (num_threads, len(dims)), the read heads.
        head1: Int32 array of shape (num_threads, len(dims)), the write heads.
        origin: Int32 array of shape (num_threads, len(dims)), where each thread
            spawned; the centre of its window when the machine has one.
        age: Int32 array of shape (num_threads,), steps since each thread spawned.
        step: Scalar int32, steps taken by the system.

    """

    memory: Array
    position: Array
    direction: Array
    head0: Array
    head1: Array
    origin: Array
    age: Array
    step: Array


class BFFGrid(ComplexSystem[GridState, Array]):
    """BFF threads on a shared memory torus of any dimension.

    One step moves every thread once, synchronously: all reads see the memory as it
    was, all writes land together, and when two threads write the same cell a random
    per-step priority decides. Every `num_steps` steps each cell mutates with
    probability `mutation_rate`, as a BFF epoch does. A thread that has lived
    `num_steps` steps respawns at a random cell with a random direction and both heads
    on its pointer.

    With a `window`, a thread's pointer and heads wrap within the `window` cells
    centred on its spawn point along every axis, and a cyclic bracket search walks
    that ring. Each thread then works in a sandbox the size of a BFF tape, as a pair
    does, except that sandboxes overlap wherever threads spawn near each other. A
    window equal to the memory is no window at all.
    """

    def __init__(
        self,
        *,
        opcode_table: Array | None = None,
        num_steps: int = 8192,
        mutation_rate: float = 1 / 4096,
        control: Control = "flip",
        turn: Turn = "reflect",
        window: int | None = None,
        anchor: int | None = None,
        headings: Headings = "both",
        heads: Heads = "absolute",
        rngs: nnx.Rngs,
    ):
        """Initialize the grid machine.

        Args:
            opcode_table: Integer array of shape (256,) mapping each byte to an `Op`;
                the ASCII encoding of the reference if None.
            num_steps: Lifetime of a thread in steps, and the mutation period.
            mutation_rate: Probability that a cell is replaced by a random byte at each
                mutation; zero disables mutation.
            control: What brackets do; see `Control`.
            turn: What `]` does when it turns under `"flip"`; see `Turn`.
            window: Side of the box around its spawn point that a thread is confined
                to, in cells, or None for the whole memory.
            headings: Which directions threads spawn with; see `Headings`.
            heads: How head moves read the heading; see `Heads`.
            anchor: If set, a thread spawns at a multiple of it along its heading, and
                a window is centred half a window ahead of the spawn point, so the
                sandbox is the stretch of memory in front of the thread, as a BFF pair
                is in front of its pointer: the memory is a weave of tapes along every
                axis, one per `anchor` cells, sharing every cell.
            rngs: Random streams. Priorities and respawns draw from `threads`,
                mutation from `mutation`; `nnx.Rngs(0)` seeds both.

        """
        if opcode_table is None:
            opcode_table = opcode_table_from_string()
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps!r}")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate!r}")
        self.opcode_table = opcode_table
        self.num_steps = num_steps
        self.mutation_rate = mutation_rate
        if window is not None and window < 2:
            raise ValueError(f"window must be at least 2, got {window!r}")
        self.control = control
        self.turn = turn
        self.window = window
        self.anchor = anchor
        self.headings = headings
        self.heads = heads
        self.rngs = rngs

    def init_state(
        self, *, shape: tuple[int, ...], num_threads: int | None = None
    ) -> GridState:
        """Create a random memory with threads scattered over it.

        Args:
            shape: Memory shape, one entry per dimension.
            num_threads: Number of threads; one per 64 cells if None.

        Returns:
            A fresh `GridState` with thread ages staggered over a lifetime.

        """
        if num_threads is None:
            num_threads = max(1, math.prod(shape) // 64)
        key_memory, key_threads = jax.random.split(self.rngs.params())
        memory = jax.random.randint(key_memory, shape, 0, 256, dtype=jnp.uint8)
        position, direction, origin = self._spawn(key_threads, num_threads, shape)
        age = jax.random.randint(
            self.rngs.params(), (num_threads,), 0, self.num_steps, dtype=jnp.int32
        )
        return GridState(
            memory=memory,
            position=position,
            direction=direction,
            head0=position,
            head1=position,
            origin=origin,
            age=age,
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def _spawn(
        self, key: Array, num: int, shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array]:
        """Random positions, directions and window centres for `num` threads."""
        key_position, key_direction = jax.random.split(key)
        d = len(shape)
        dims = jnp.asarray(shape, dtype=jnp.int32)
        position = jax.random.randint(key_position, (num, d), 0, dims)
        count = d if self.headings == "positive" else 2 * d
        direction = jax.random.randint(key_direction, (num,), 0, count)
        along = jax.nn.one_hot(direction % d, d, dtype=jnp.int32)
        origin = position
        if self.anchor is not None:
            # Along the heading the start is a multiple of the anchor; across it, any.
            position = jnp.where(along > 0, position - position % self.anchor, position)
            if self.window is not None:
                sign = jnp.where(direction < d, 1, -1)[:, None]
                origin = (position + (self.window // 2) * along * sign) % dims
        return position.astype(jnp.int32), direction.astype(jnp.int32), origin

    def _confine(self, index: Array, origin: Array, dims: Array) -> Array:
        """Wrap positions onto the torus, and into the thread's window if it has one."""
        if self.window is None:
            return index % dims
        half = self.window // 2
        return (origin + (index - origin + half) % self.window - half) % dims

    @override
    def _step(self, state: GridState, input: Array | None = None) -> GridState:
        memory = state.memory
        shape = memory.shape
        d = len(shape)
        dims = jnp.asarray(shape, dtype=jnp.int32)
        num = state.position.shape[0]
        origin = state.origin
        key_priority, key_spawn = jax.random.split(self.rngs.threads())

        def at(index: Array) -> tuple[Array, ...]:
            return tuple(index[:, k] for k in range(d))

        cmd = self.opcode_table[memory[at(state.position)]]
        value0 = memory[at(state.head0)]
        value1 = memory[at(state.head1)]

        # Writes: at most one byte per thread; conflicts go to a random priority.
        target = jnp.where((cmd == Op.COPY01)[:, None], state.head1, state.head0)
        value = jnp.select(
            [cmd == Op.PLUS, cmd == Op.MINUS, cmd == Op.COPY01, cmd == Op.COPY10],
            [value0 + 1, value0 - 1, value0, value1],
            default=memory[at(target)],
        ).astype(jnp.uint8)
        writes = (cmd >= Op.PLUS) & (cmd <= Op.COPY10)
        priority = jax.random.randint(key_priority, (num,), 0, 1 << 16)
        entry = jnp.where(writes, (priority << 8) | value.astype(jnp.int32), -1)
        winner = jnp.full(shape, -1, dtype=jnp.int32).at[at(target)].max(entry)
        memory = jnp.where(winner >= 0, (winner & 255).astype(jnp.uint8), memory)

        # Heads move along the thread's axis; brackets turn; the pointer advances.
        def count(*ops: Op) -> Array:
            return sum((cmd == op).astype(jnp.int32) for op in ops)

        along = jax.nn.one_hot(state.direction % d, d, dtype=jnp.int32)
        stride = along
        if self.heads == "relative":
            stride = along * jnp.where(state.direction < d, 1, -1)[:, None]
        head0 = state.head0 + stride * (count(Op.INC0) - count(Op.DEC0))[:, None]
        head1 = state.head1 + stride * (count(Op.INC1) - count(Op.DEC1))[:, None]
        ahead = (cmd == Op.LOOP_START) & (value0 == 0)
        back = (cmd == Op.LOOP_END) & (value0 != 0)
        direction = state.direction
        position = state.position
        if self.control == "flip":
            close = 1 if self.turn == "quarter" else d
            turn = close * back.astype(jnp.int32) - ahead.astype(jnp.int32)
            direction = (direction + turn) % (2 * d)
        else:
            sign = jnp.where(direction < d, 1, -1)
            step = sign * jnp.where(back, -1, 1)
            target = self._match(memory, position, origin, along, step, ahead)
            position = jnp.where((ahead | back)[:, None], target, position)
        sign = jnp.where(direction < d, 1, -1)[:, None]
        along = jax.nn.one_hot(direction % d, d, dtype=jnp.int32)
        position = position + along * sign

        # Threads at the end of their life respawn with the heads on the pointer.
        age = state.age + 1
        respawn = age >= self.num_steps
        spawn_position, spawn_direction, spawn_origin = self._spawn(
            key_spawn, num, shape
        )
        confined = self._confine(position, origin, dims)
        position = jnp.where(respawn[:, None], spawn_position, confined)
        direction = jnp.where(respawn, spawn_direction, direction)
        head0 = jnp.where(
            respawn[:, None], position, self._confine(head0, origin, dims)
        )
        head1 = jnp.where(
            respawn[:, None], position, self._confine(head1, origin, dims)
        )
        origin = jnp.where(respawn[:, None], spawn_origin, origin)
        age = jnp.where(respawn, 0, age)

        step = state.step + 1
        if self.mutation_rate > 0.0:
            key = self.rngs.mutation()

            def mutate(memory: Array) -> Array:
                key_mask, key_byte = jax.random.split(key)
                mask = jax.random.uniform(key_mask, shape) < self.mutation_rate
                byte = jax.random.randint(key_byte, shape, 0, 256).astype(jnp.uint8)
                return jnp.where(mask, byte, memory)

            memory = jax.lax.cond(
                step % self.num_steps == 0, mutate, lambda memory: memory, memory
            )
        return GridState(
            memory=memory,
            position=position,
            direction=direction,
            head0=head0,
            head1=head1,
            origin=origin,
            age=age,
            step=step,
        )

    def _match(
        self,
        memory: Array,
        position: Array,
        origin: Array,
        along: Array,
        sign: Array,
        opening_bracket: Array,
    ) -> Array:
        """Position of the bracket matching the one under each thread, along its line.

        Walks `sign` cells at a time from each thread's position along the axis given by
        `along`, wrapping on the torus or within the window, counting the partner kind
        of bracket as -1 and the thread's own kind as +1 from 1 (`opening_bracket` says
        which is which), and stops where the count reaches zero; a thread whose walk
        returns to its start keeps its position.
        """
        shape = memory.shape
        d = len(shape)
        dims = jnp.asarray(shape, dtype=jnp.int32)
        if self.window is None:
            length = (along * dims).sum(axis=-1)
        else:
            length = jnp.full(position.shape[0], self.window, dtype=jnp.int32)
        offset = jnp.arange(self.window or max(shape), dtype=jnp.int32)
        walked = (
            position[:, None, :]
            + (along * sign[:, None])[:, None, :] * offset[None, :, None]
        )
        walked = self._confine(walked, origin[:, None, :], dims)
        ops = self.opcode_table[memory[tuple(walked[..., k] for k in range(d))]]
        opening = (ops == Op.LOOP_START).astype(jnp.int32)
        closing = (ops == Op.LOOP_END).astype(jnp.int32)
        delta = jnp.where(
            opening_bracket[:, None], opening - closing, closing - opening
        )
        delta = jnp.where(offset[None, :] < length[:, None], delta, 0).at[:, 0].set(1)
        total = jnp.cumsum(delta, axis=-1)
        candidate = (
            (offset[None, :] > 0) & (offset[None, :] < length[:, None]) & (total == 0)
        )
        first = jnp.argmax(candidate, axis=-1)
        found = jnp.any(candidate, axis=-1)
        target = jnp.take_along_axis(walked, first[:, None, None], axis=1)[:, 0, :]
        return jnp.where(found[:, None], target, position)

    @nnx.jit
    @override
    def render(self, state: GridState) -> Array:
        """Render the memory as an image with the threads overlaid.

        Each of the ten instructions has its own hue, data bytes are greys by value,
        the zero byte is near black, and each thread is a white pixel at its pointer.
        A one-dimensional memory renders as a single row; with more than two
        dimensions the middle slice along the first axis is shown.

        Args:
            state: Grid state.

        Returns:
            RGB image with dtype uint8 and shape (height, width, 3).

        """
        memory, position = state.memory, state.position
        visible = jnp.ones(position.shape[0], dtype=bool)
        while memory.ndim > 2:
            middle = memory.shape[0] // 2
            visible &= position[:, 0] == middle
            memory, position = memory[middle], position[:, 1:]
        if memory.ndim == 1:
            memory = memory[None]
            position = jnp.concatenate([jnp.zeros_like(position), position], axis=-1)
        byte = jnp.arange(256, dtype=jnp.int32)
        grey = (48 + (byte * 3) // 4).astype(jnp.uint8)
        palette = jnp.stack([grey, grey, grey], axis=-1)
        colours = jnp.array(
            [
                (0, 200, 90),  # [
                (0, 140, 60),  # ]
                (255, 90, 90),  # +
                (200, 40, 40),  # -
                (255, 60, 220),  # .
                (170, 30, 170),  # ,
                (90, 150, 255),  # <
                (40, 100, 230),  # >
                (250, 200, 60),  # {
                (220, 150, 20),  # }
                (12, 12, 12),  # zero byte
            ],
            dtype=jnp.uint8,
        )
        op = self.opcode_table
        palette = jnp.where(
            (op <= Op.NULL)[:, None], colours[jnp.minimum(op, 10)], palette
        )
        image = palette[memory.astype(jnp.int32)]
        white = jnp.where(visible, 255, 0).astype(jnp.uint8)[:, None]
        return image.at[position[:, 0], position[:, 1]].max(white)
