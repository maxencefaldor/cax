"""BFF module.

This module implements the BFF primordial soup of Agüera y Arcas et al. (2024): a
population of short random byte strings, each both a program and its own memory,
interpreted in a Brainfuck dialect whose input and output streams are replaced by copy
instructions between two heads on the same tape. There is no fitness function. Each
epoch, programs are paired at random, each pair is concatenated into one tape, mutated,
executed for a fixed step budget, and split back into the soup. Self-replicators emerge
from this interaction alone and, once they do, the soup's compressibility collapses.

The system is not a cellular automaton: it has no neighbourhood, and an epoch is a
global random pairing. It sits in the zoo for the same reason Boids and Particle Life
do: one state, one step, everything vectorised.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108
    [2] BFF: Simple explanations for complex phenomena, Knierim et al. 2026.
        https://arxiv.org/abs/2607.01483
    [3] cubff reference implementation.
        https://github.com/paradigms-of-intelligence/cubff

"""

from functools import partial
from typing import override

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.sharding import PartitionSpec as P

from cax.core import ComplexSystem

from .interpreter import Control, Implementation, run
from .language import Op, opcode_table_from_string


class BFF(ComplexSystem[Array, Array]):
    """BFF primordial soup.

    The state is the soup: an unsigned 8-bit array of shape
    `(num_programs, tape_length)`. One step is one epoch: a uniformly random pairing of
    all programs, per-byte background mutation of every pair, execution of each pair as
    a single tape of twice the length, and the write-back of both halves.

    The reference implementation's languages are configurations of this class:
    `bff_noheads` (the paper's Section 2, and its headline result) is the default;
    `bff` is `heads_from_tape=True`; `bff_perm` is `opcode_table_permuted()`.

    Speed: on a GPU each pair runs inside one Pallas kernel that stops when the tape
    halts, as the reference does (see `run`); an epoch of the paper's 2^17-program soup
    takes about 20 ms on an H100 against 12 ms for the reference's CUDA build. On CPU
    the XLA scan pays for the full step budget of every tape and a laptop takes about
    4 s per epoch against 0.13 s for the reference's single-threaded build. What this
    class offers is the rest of the ecosystem: the same state and step API as every
    other system, and states that are ordinary arrays.

    Sharding: with `shard_axis` set to the name of an axis of the active mesh (see
    `jax.set_mesh`), `init_state` places the soup along that axis and each epoch runs
    the pairs device-locally under `jax.shard_map`, leaving the random pairing's
    gather and scatter to the compiler. One epoch is bit-identical to the unsharded
    one. A soup of 2^17 programs is not enough work to keep one GPU busy, so this
    pays only for soups of millions of programs; for many independent soups run
    independent processes.
    """

    def __init__(
        self,
        *,
        opcode_table: Array | None = None,
        num_steps: int = 8192,
        heads_from_tape: bool = False,
        control: Control = "matched",
        mutation_rate: float = 1 / 4096,
        implementation: Implementation = "auto",
        shard_axis: str | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize BFF.

        Args:
            opcode_table: Integer array of shape (256,) mapping each byte to an `Op`;
                the ASCII encoding `[]+-.,<>{}` of the reference if None.
            num_steps: Step budget per pair execution. The reference uses 8192.
            heads_from_tape: If False, the instruction pointer and both heads start at
                zero (`bff_noheads`). If True, the heads start at the first two bytes of
                the tape and the instruction pointer at 2 (`bff`).
            control: Control-flow rule: `"matched"` is the reference; `"cyclic"` and
                `"flip"` are the total, never-halting variants described in the
                interpreter module.
            mutation_rate: Probability that each byte of a concatenated pair is replaced
                by a uniformly random byte before execution. The reference default
                1/4096 is the paper's 0.024%. Zero disables mutation.
            implementation: Interpreter implementation passed to `run`.
            shard_axis: Name of the mesh axis to shard the soup over, or None to run
                on one device.
            rngs: Random streams. Pairing draws from `pairing`, mutation from
                `mutation`; `nnx.Rngs(0)` seeds both from the default stream.

        """
        if opcode_table is None:
            opcode_table = opcode_table_from_string()
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps!r}")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate!r}")
        self.opcode_table = opcode_table
        self.num_steps = num_steps
        self.heads_from_tape = heads_from_tape
        self.control = control
        self.mutation_rate = mutation_rate
        self.implementation = implementation
        self.shard_axis = shard_axis
        self.rngs = rngs

    def init_state(self, *, num_programs: int, tape_length: int = 64) -> Array:
        """Create a uniformly random soup.

        Args:
            num_programs: Number of programs; must be even so every program is paired.
            tape_length: Bytes per program. The reference uses 64.

        Returns:
            Unsigned 8-bit array of shape (num_programs, tape_length).

        """
        if num_programs % 2 != 0:
            raise ValueError(f"num_programs must be even, got {num_programs!r}")
        soup = jax.random.randint(
            self.rngs.params(), (num_programs, tape_length), 0, 256, dtype=jnp.uint8
        )
        if self.shard_axis is not None:
            soup = jax.device_put(soup, P(self.shard_axis))
        return soup

    def pair_and_run(self, soup: Array, permutation: Array) -> tuple[Array, Array]:
        """Execute one epoch given the pairing.

        Args:
            soup: Unsigned 8-bit array of shape (num_programs, tape_length).
            permutation: Integer array of shape (num_programs,); program
                `permutation[2i]` is the first half of pair `i`, `permutation[2i + 1]`
                the second.

        Returns:
            A tuple `(soup, steps)`: the soup after execution, and an int32 array of
                shape (num_programs // 2,) with the number of steps each pair ran.

        """
        num_programs, tape_length = soup.shape
        spec = P(self.shard_axis) if self.shard_axis is not None else None

        def constrain(x: Array) -> Array:
            return x if spec is None else jax.lax.with_sharding_constraint(x, spec)

        pairs = constrain(soup[permutation]).reshape(num_programs // 2, 2 * tape_length)
        if self.mutation_rate > 0.0:
            key = self.rngs.mutation()
            key_mask, key_byte = jax.random.split(key)
            mutate = jax.random.uniform(key_mask, pairs.shape) < self.mutation_rate
            replacement = jax.random.randint(key_byte, pairs.shape, 0, 256).astype(
                jnp.uint8
            )
            pairs = jnp.where(mutate, replacement, pairs)
        execute = partial(
            run,
            num_steps=self.num_steps,
            heads_from_tape=self.heads_from_tape,
            control=self.control,
            implementation=self.implementation,
        )
        if spec is not None:
            pairs = constrain(pairs)
            # check_vma is off because a Pallas call's outputs carry no varying-axis
            # metadata; the specs above say everything the partitioner needs.
            execute = jax.shard_map(
                execute,
                in_specs=(spec, P()),
                out_specs=(spec, spec, spec),
                check_vma=False,
            )
        pairs, steps, _ = execute(pairs, self.opcode_table)
        # Write back through the inverse permutation: a gather, which partitions as an
        # all-gather plus a local gather, where the equivalent scatter does not.
        inverse = (
            jnp.zeros_like(permutation)
            .at[permutation]
            .set(jnp.arange(num_programs, dtype=permutation.dtype))
        )
        soup = constrain(constrain(pairs.reshape(soup.shape))[inverse])
        return soup, steps

    @override
    def _step(self, state: Array, input: Array | None = None) -> Array:
        num_programs = state.shape[0]
        permutation = jax.random.permutation(self.rngs.pairing(), num_programs)
        soup, _ = self.pair_and_run(state, permutation)
        return soup

    @nnx.jit
    @override
    def render(self, state: Array) -> Array:
        """Render a soup to an RGB image, one row per program, one pixel per byte.

        Colours follow the reference visualiser: brackets green, the four tape
        writes magenta, head moves lilac, the zero byte red, and every other byte a grey
        whose brightness encodes its value.

        Args:
            state: Unsigned 8-bit array of shape (num_programs, tape_length).

        Returns:
            RGB image with dtype uint8 and shape (num_programs, tape_length, 3).

        """
        byte = jnp.arange(256, dtype=jnp.int32)
        grey = (192 + byte // 4).astype(jnp.uint8)
        palette = jnp.stack([grey, grey, grey], axis=-1)
        op = self.opcode_table
        colours = [
            ((op == Op.LOOP_START) | (op == Op.LOOP_END), (0, 192, 0)),
            (
                (op == Op.PLUS)
                | (op == Op.MINUS)
                | (op == Op.COPY01)
                | (op == Op.COPY10),
                (200, 0, 200),
            ),
            (
                (op == Op.DEC0) | (op == Op.INC0) | (op == Op.DEC1) | (op == Op.INC1),
                (200, 128, 220),
            ),
            (op == Op.NULL, (255, 0, 0)),
        ]
        for mask, rgb in colours:
            palette = jnp.where(
                mask[:, None], jnp.array(rgb, dtype=jnp.uint8)[None, :], palette
            )
        return palette[state.astype(jnp.int32)]
