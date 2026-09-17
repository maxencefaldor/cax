"""Tests for BFF."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cax.cs.bff import (
    BFF,
    Op,
    byte_entropy,
    high_order_entropy,
    initial_thread_state,
    is_instruction,
    local_pairing,
    opcode_table_from_string,
    opcode_table_permuted,
    opcode_table_swap_heads,
    parse,
    run,
    skeleton_hash,
    step,
    unparse,
)

TABLE = opcode_table_from_string()


def tape_from(program: str, length: int = 128) -> jnp.ndarray:
    """Parse a program and zero-pad it to a full tape."""
    bytes_ = parse(program)
    return jnp.zeros((length,), dtype=jnp.uint8).at[: bytes_.shape[0]].set(bytes_)


def test_bff_jit_init() -> None:
    """Test that BFF can be instantiated under jax.jit."""

    @jax.jit
    def init_bff() -> BFF:
        return BFF(rngs=nnx.Rngs(0))

    try:
        init_bff()
    except Exception as e:
        pytest.fail(f"BFF instantiation failed under jit: {e}")


def test_opcode_table_ascii() -> None:
    assert int(TABLE[ord("[")]) == Op.LOOP_START
    assert int(TABLE[ord("}")]) == Op.INC1
    assert int(TABLE[0]) == Op.NULL
    assert int(TABLE[ord("a")]) == Op.NOOP
    assert int((TABLE < Op.NULL).sum()) == 10


def test_opcode_table_permuted() -> None:
    table = opcode_table_permuted()
    assert int(table[0]) == Op.NULL
    assert [int(table[b]) for b in range(1, 11)] == [
        Op.DEC0,
        Op.INC0,
        Op.DEC1,
        Op.INC1,
        Op.PLUS,
        Op.MINUS,
        Op.COPY01,
        Op.COPY10,
        Op.LOOP_START,
        Op.LOOP_END,
    ]


def test_parse_unparse_roundtrip() -> None:
    program = "[[{.>]-]]-]>.{[[0"
    assert unparse(parse(program)) == program


def test_heads_from_tape() -> None:
    tape = jnp.zeros((128,), dtype=jnp.uint8).at[0].set(200).at[1].set(5)
    state = initial_thread_state(tape, heads_from_tape=True)
    assert int(state.pc) == 2
    assert int(state.head0) == 200 % 128
    assert int(state.head1) == 5
    state = initial_thread_state(tape, heads_from_tape=False)
    assert int(state.pc) == 0
    assert int(state.head0) == 0
    assert int(state.head1) == 0


def test_step_minus_is_self_modification() -> None:
    tape = tape_from("-")
    state = step(initial_thread_state(tape, heads_from_tape=False), TABLE)
    # head0 is at 0, which holds the `-` byte itself.
    assert int(state.tape[0]) == ord("-") - 1
    assert int(state.pc) == 1
    assert not bool(state.halted)


def test_step_head_wraps_tape() -> None:
    tape = tape_from("<")
    state = step(initial_thread_state(tape, heads_from_tape=False), TABLE)
    assert int(state.head0) == 127


def test_run_halts_off_the_end() -> None:
    tapes = jnp.zeros((1, 128), dtype=jnp.uint8)
    _, steps, ops, _ = run(tapes, TABLE, num_steps=8192)
    assert int(steps[0]) == 128
    assert int(ops[0]) == 0


def test_run_unmatched_bracket_halts() -> None:
    # `[` with zero at head0 and no `]` anywhere: the taken jump fails, halt.
    tapes = tape_from("0[")[None]
    _, steps, _, _ = run(tapes, TABLE, num_steps=8192)
    assert int(steps[0]) == 2
    # `]` with nonzero at head0 (the `]` byte itself) and no `[`: halt.
    tapes = tape_from("]")[None]
    _, steps, _, _ = run(tapes, TABLE, num_steps=8192)
    assert int(steps[0]) == 1
    # An untaken jump never scans: `[` with nonzero at head0 falls through.
    tapes = tape_from("[")[None]
    _, steps, _, _ = run(tapes, TABLE, num_steps=8192)
    assert int(steps[0]) == 128


def test_run_infinite_loop_uses_budget() -> None:
    # `+[]`: tape[0] becomes nonzero, `[` falls through, `]` jumps back, forever.
    tapes = tape_from("+[]")[None]
    _, steps, ops, _ = run(tapes, TABLE, num_steps=1000)
    assert int(steps[0]) == 1000
    assert int(ops[0]) == 1000


def test_run_forward_skip_nested() -> None:
    # tape[0] is the zero byte, so `[` is taken and must skip past the matching outer
    # `]`, with nesting, and only then execute the final `+` on tape[0].
    tapes = tape_from("0[[+]]+")[None]
    out, steps, _, _ = run(tapes, TABLE, num_steps=8192)
    assert int(out[0, 0]) == 1
    # The jump lands on the outer `]`; positions 2 to 4 are never visited.
    assert int(steps[0]) == 124


def test_run_matches_step() -> None:
    """The batched fast path and the single-thread reference agree on random tapes."""
    rng = np.random.default_rng(0)
    ops = np.frombuffer(b"[]+-.,<>{}", dtype=np.uint8)
    tapes = rng.integers(0, 256, (256, 128), dtype=np.uint8)
    mask = rng.random(tapes.shape)
    tapes = np.where(mask < 0.4, rng.choice(ops, tapes.shape), tapes)
    tapes = np.where((mask >= 0.4) & (mask < 0.5), 0, tapes).astype(np.uint8)
    tapes = jnp.asarray(tapes)

    def reference(tape):
        state = initial_thread_state(tape, heads_from_tape=True)
        state = jax.lax.fori_loop(0, 300, lambda _, s: step(s, TABLE), state)
        return state.tape, state.steps, state.ops

    ref_tape, ref_steps, ref_ops = jax.vmap(reference)(tapes)
    for kwargs in (
        {},
        {"scan_fraction": 1.0, "compact_after": 300},
        {"scan_fraction": 1 / 256, "compact_after": 10, "compact_fraction": 1 / 256},
    ):
        out_tape, out_steps, out_ops, _ = run(
            tapes, TABLE, num_steps=300, heads_from_tape=True, **kwargs
        )
        assert bool(jnp.all(out_tape == ref_tape))
        assert bool(jnp.all(out_steps == ref_steps))
        assert bool(jnp.all(out_ops == ref_ops))


def test_paper_replicator_copies_itself() -> None:
    """The replicator of Figure 4 copies itself into a zeroed second half.

    The program is a palindrome, so copying it in reverse reproduces it.
    """
    program = "[[{.>]-]]-]>.{[["
    tapes = tape_from(program)[None]
    out, steps, _, _ = run(tapes, TABLE, num_steps=8192, heads_from_tape=False)
    assert int(steps[0]) == 8192
    assert program in unparse(out[0, 64:])


def test_bff_step_shape_and_dtype() -> None:
    cs = BFF(num_steps=1, mutation_rate=0.0, rngs=nnx.Rngs(0))
    soup = cs.init_state(num_programs=64)
    next_soup = cs(soup, num_steps=1)
    assert next_soup.shape == soup.shape
    assert next_soup.dtype == jnp.uint8
    # One instruction per pair: at most one byte changes per pair.
    assert int((next_soup != soup).sum()) <= 32


def test_bff_pair_and_run_keeps_slots() -> None:
    cs = BFF(num_steps=1, mutation_rate=0.0, rngs=nnx.Rngs(0))
    soup = cs.init_state(num_programs=16)
    out, steps, _ = cs.pair_and_run(soup, jnp.arange(16))
    assert steps.shape == (8,)
    assert int((out != soup).sum()) <= 8


def test_bff_mutation_rate_one_resamples() -> None:
    cs = BFF(num_steps=1, mutation_rate=1.0, rngs=nnx.Rngs(0))
    soup = jnp.zeros((16, 64), dtype=jnp.uint8)
    out = cs(soup, num_steps=1)
    assert int((out != 0).sum()) > 16 * 64 // 2


def test_bff_render() -> None:
    cs = BFF(rngs=nnx.Rngs(0))
    soup = cs.init_state(num_programs=8)
    rgb = cs.render(soup)
    assert rgb.shape == (8, 64, 3)
    assert rgb.dtype == jnp.uint8
    zero = cs.render(jnp.zeros((1, 1), jnp.uint8))[0, 0]
    assert tuple(int(v) for v in zero) == (255, 0, 0)


def test_metrics_random_versus_repeated() -> None:
    rng = np.random.default_rng(0)
    random_soup = rng.integers(0, 256, (1024, 64), dtype=np.uint8)
    repeated = np.tile(random_soup[:1], (1024, 1))
    assert abs(high_order_entropy(random_soup, compressor="zlib")) < 0.5
    assert high_order_entropy(repeated, compressor="zlib") > 5.0
    assert abs(byte_entropy(random_soup) - 8.0) < 0.05


def test_replication_score_of_paper_replicator() -> None:
    from cax.cs.bff import replication_score, sample_partners

    program = parse("[[{.>]-]A]-]>.{[[")
    programs = (
        jnp.zeros((2, 64), dtype=jnp.uint8).at[0, : program.shape[0]].set(program)
    )
    programs = programs.at[1].set(
        jax.random.randint(jax.random.key(1), (64,), 0, 256, dtype=jnp.uint8)
    )
    partners = sample_partners(jax.random.key(0), 2)
    scores = replication_score(programs, partners, TABLE)
    # The replicator reproduces its 17 bytes in every chain; a random program
    # reproduces nothing. The score counts bytes, so it is bounded by the length.
    assert int(scores[0]) == 17
    assert int(scores[1]) == 0
    # Under the other head convention it does not run at all.
    scores = replication_score(programs, partners, TABLE, heads_from_tape=True)
    assert int(scores[0]) == 0


def test_score_from_tapes_counts_agreeing_positions() -> None:
    from cax.cs.bff import NUM_CHAINS, score_from_tapes

    program = jnp.arange(64, dtype=jnp.uint8)
    tapes = jnp.tile(jnp.arange(128, dtype=jnp.uint8), (NUM_CHAINS, 1))
    # Every chain agrees everywhere and the first half equals the program: 64.
    assert int(score_from_tapes(program, tapes)) == 64
    # Break the first half of every chain at one position: 63.
    tapes = tapes.at[:, 3].set(255)
    assert int(score_from_tapes(program, tapes)) == 63
    # Agreement in only three chains is not enough (needs more than 13 // 4).
    tapes = jnp.tile(jnp.arange(128, dtype=jnp.uint8), (NUM_CHAINS, 1))
    tapes = tapes.at[3:, 70].set(jnp.arange(10, dtype=jnp.uint8) + 200)
    assert int(score_from_tapes(program, tapes)) == 63


def test_flip_walls_hold_a_pointer_that_starts_inside() -> None:
    """Under `flip`, `]+]` is a room while the test byte is nonzero.

    `]` reflects on both faces, so a pointer starting on the `+` bounces between the
    walls, incrementing the test byte on every other step, until the byte wraps to zero
    and the walls open.

    Heads are seeded from the tape so that head0 sits on a data cell (100) rather than
    on the `+` byte itself, which would otherwise rewrite the instruction, and so that
    the pointer starts at 2, on the `+`.
    """
    tape = jnp.zeros((128,), dtype=jnp.uint8).at[0].set(100).at[100].set(1)
    tape = tape.at[1:4].set(parse("]+]"))
    out, steps, ops, _ = run(
        tape[None], TABLE, num_steps=300, control="flip", heads_from_tape=True
    )
    assert int(steps[0]) == 300
    assert int(ops[0]) == 300
    assert 140 <= int(out[0, 100]) <= 160


def test_flip_never_halts_and_wraps() -> None:
    tapes = jnp.zeros((1, 128), dtype=jnp.uint8)
    _, steps, _, _ = run(tapes, TABLE, num_steps=1000, control="flip")
    assert int(steps[0]) == 1000


def test_cyclic_unmatched_is_noop_and_wraps() -> None:
    # `0[` : the taken `[` has no partner anywhere; under `cyclic` it falls through
    # and the pointer keeps circling the tape for the whole budget.
    tapes = tape_from("0[")[None]
    _, steps, _, _ = run(tapes, TABLE, num_steps=1000, control="cyclic")
    assert int(steps[0]) == 1000


def test_cyclic_matches_across_the_seam() -> None:
    # `]` at 0 with nonzero test cell and its `[` at the very end of the tape.
    tape = jnp.zeros((128,), dtype=jnp.uint8)
    tape = tape.at[0].set(ord("]")).at[127].set(ord("[")).at[1].set(ord("+"))
    from cax.cs.bff import initial_thread_state, match_bracket_cyclic

    assert int(match_bracket_cyclic(tape, jnp.int32(0), jnp.array(False), TABLE)) == 127
    state = initial_thread_state(tape, heads_from_tape=False)
    # tape[0] is `]` (nonzero): jump to 127, then advance past it to 0: a 2-cycle.
    state = step(state, TABLE, control="cyclic")
    assert int(state.pc) == 0
    assert not bool(state.halted)


def test_variants_fast_path_matches_step() -> None:
    from cax.cs.bff import initial_thread_state, opcode_table_swap_heads

    rng = np.random.default_rng(1)
    for table in (TABLE, opcode_table_swap_heads()):
        ops = np.array(
            [b for b in range(256) if int(table[b]) != Op.NOOP][1:], dtype=np.uint8
        )
        tapes = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        mask = rng.random(tapes.shape)
        tapes = np.where(mask < 0.4, rng.choice(ops, tapes.shape), tapes)
        tapes = jnp.asarray(
            np.where((mask >= 0.4) & (mask < 0.5), 0, tapes).astype(np.uint8)
        )
        for control in ("matched", "cyclic", "flip"):

            def reference(tape, control=control, table=table):
                state = initial_thread_state(tape, heads_from_tape=False)
                state = jax.lax.fori_loop(
                    0, 200, lambda _, s: step(s, table, control=control), state
                )
                return state.tape, state.steps, state.ops

            ref = jax.vmap(reference)(tapes)
            out = run(
                tapes, table, num_steps=200, control=control, scan_fraction=1 / 32
            )
            for a, b in zip(out, ref, strict=True):
                assert bool(jnp.all(a == b)), control


def test_swap_heads_copy_idiom() -> None:
    """With the swap dialect, `.` copies head0 to head1 and `~` exchanges them."""
    from cax.cs.bff import opcode_table_swap_heads

    table = opcode_table_swap_heads()
    # `>~.`: head0 -> 1, swap (head0 = 0, head1 = 1), copy tape[0] -> tape[1].
    tapes = tape_from(">~.", 128)[None]
    tapes = jnp.asarray(parse(">~.", opcode_table=table))
    tapes = jnp.zeros((1, 128), dtype=jnp.uint8).at[0, :3].set(tapes)
    out, _, ops, _ = run(tapes, table, num_steps=10)
    assert int(out[0, 1]) == int(out[0, 0]) == ord(">")
    assert int(ops[0]) == 3


def _enriched_tapes(rng: np.random.Generator, table, num: int) -> jnp.ndarray:
    ops = np.array(
        [b for b in range(256) if int(table[b]) != Op.NOOP][1:], dtype=np.uint8
    )
    tapes = rng.integers(0, 256, (num, 128), dtype=np.uint8)
    mask = rng.random(tapes.shape)
    tapes = np.where(mask < 0.4, rng.choice(ops, tapes.shape), tapes)
    return jnp.asarray(
        np.where((mask >= 0.4) & (mask < 0.5), 0, tapes).astype(np.uint8)
    )


@pytest.mark.parametrize("control", ["matched", "cyclic", "flip"])
@pytest.mark.parametrize("heads", [False, True])
def test_kernel_matches_xla(control: str, heads: bool) -> None:
    """The Pallas kernel computes the same function as the XLA scan."""
    from cax.cs.bff.kernel import run_kernel

    interpret = jax.default_backend() != "gpu"
    rng = np.random.default_rng(2)
    for table in (TABLE, opcode_table_swap_heads()):
        tapes = _enriched_tapes(rng, table, 70)  # not a multiple of the block
        ref = run(
            tapes,
            table,
            num_steps=300,
            heads_from_tape=heads,
            control=control,
            implementation="xla",
        )
        out = run_kernel(
            tapes,
            table,
            num_steps=300,
            heads_from_tape=heads,
            control=control,
            interpret=interpret,
        )
        for a, b in zip(out, ref, strict=True):
            assert bool(jnp.all(a == b)), control


@pytest.mark.skipif(jax.device_count() < 2, reason="needs several devices")
def test_bff_sharded_epoch_matches_unsharded() -> None:
    """An epoch of a soup sharded over a mesh axis is bit-identical to the plain one."""
    from jax.sharding import AxisType

    mesh = jax.make_mesh(
        (jax.device_count(),), ("program",), axis_types=(AxisType.Auto,)
    )

    def epoch(cs: BFF, soup: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        permutation = jax.random.permutation(jax.random.key(5), soup.shape[0])
        out, steps, _ = cs.pair_and_run(soup, permutation)
        return out, steps

    with jax.set_mesh(mesh):
        plain = BFF(num_steps=300, rngs=nnx.Rngs(0))
        sharded = BFF(num_steps=300, shard_axis="program", rngs=nnx.Rngs(0))
        soup = plain.init_state(num_programs=64 * jax.device_count())
        soup_sharded = sharded.init_state(num_programs=64 * jax.device_count())
        assert bool(jnp.all(soup == soup_sharded))
        out, steps = nnx.jit(epoch)(plain, soup)
        out_sharded, steps_sharded = nnx.jit(epoch)(sharded, soup_sharded)
    assert bool(jnp.all(out == out_sharded))
    assert bool(jnp.all(steps == steps_sharded))


def test_bff_batched_soups_match_single() -> None:
    """A batch of soups steps exactly as each soup would alone (no mutation)."""
    cs = BFF(num_steps=200, mutation_rate=0.0, rngs=nnx.Rngs(0))
    soups = cs.init_state(num_programs=64, num_soups=3)
    keys = jax.random.split(jax.random.key(9), 3)
    permutations = jax.vmap(lambda k: jax.random.permutation(k, 64))(keys)
    batched, steps, _ = cs.pair_and_run(soups, permutations)
    assert batched.shape == soups.shape
    assert steps.shape == (3, 32)
    for i in range(3):
        single, single_steps, _ = cs.pair_and_run(soups[i], permutations[i])
        assert bool(jnp.all(single == batched[i]))
        assert bool(jnp.all(single_steps == steps[i]))
    stepped = cs(soups)
    assert stepped.shape == soups.shape


def test_local_pairing_is_a_perfect_matching_of_neighbours() -> None:
    height, width = 6, 8
    for seed in range(8):
        pairs = np.asarray(local_pairing(jax.random.key(seed), height, width))
        assert sorted(pairs.tolist()) == list(range(height * width))
        pairs = pairs.reshape(-1, 2)
        dy = (pairs[:, 0] // width - pairs[:, 1] // width) % height
        dx = (pairs[:, 0] % width - pairs[:, 1] % width) % width
        assert np.all((dy % (height - 2) == 1) ^ (dx % (width - 2) == 1))
        assert np.all((dy == 0) | (dx == 0))


def test_bff_grid_step_and_render() -> None:
    cs = BFF(num_steps=1, mutation_rate=0.0, grid=(4, 6), rngs=nnx.Rngs(0))
    soup = cs.init_state()
    assert soup.shape == (24, 64)
    assert cs(soup, num_steps=2).shape == soup.shape
    rgb = cs.render(soup)
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == jnp.uint8
    with pytest.raises(ValueError):
        cs.init_state(num_programs=16)
    with pytest.raises(ValueError):
        BFF(grid=(3, 4), rngs=nnx.Rngs(0))


def test_skeleton_hash_ignores_data_bytes() -> None:
    cs = BFF(rngs=nnx.Rngs(0))
    soup = cs.init_state(num_programs=4)
    op = cs.opcode_table[soup.astype(jnp.int32)]
    data = ~is_instruction(op)
    noise = jax.random.randint(jax.random.key(1), soup.shape, 0, 256).astype(jnp.uint8)
    noise = jnp.where(cs.opcode_table[noise.astype(jnp.int32)] == Op.NOOP, noise, 1)
    same = jnp.where(data, noise, soup)
    flipped = soup.at[:, 0].set(ord("["))
    hashes = [skeleton_hash(s, cs.opcode_table) for s in (soup, same, flipped)]
    assert bool((hashes[0] == hashes[1]).all())
    assert bool((hashes[0] != hashes[2]).all())


def test_bff_energy_economy() -> None:
    from cax.cs.bff import BFFEconomy, EconomyState

    # All-zero tapes: every pair runs its 64 steps in the first half, so with no
    # income the first program of each pair dies and is replaced by random bytes.
    cs = BFFEconomy(
        income=0, bff=BFF(num_steps=64, mutation_rate=0.0, rngs=nnx.Rngs(0))
    )
    state = cs.init_state(num_programs=16)
    assert state.energy.shape == (16,)
    state = EconomyState(soup=jnp.zeros((16, 64), jnp.uint8), energy=state.energy)
    out = cs(state, num_steps=1)
    assert bool((out.energy == 0).all())
    assert int((out.soup != 0).any(axis=-1).sum()) == 8
    assert cs.render(out).shape == (16, 64, 3)
    # With an income covering the budget nobody dies and energy only accumulates.
    rich = BFFEconomy(
        income=64, bff=BFF(num_steps=64, mutation_rate=0.0, rngs=nnx.Rngs(0))
    )
    out = rich(rich.init_state(num_programs=16), num_steps=2)
    assert bool((out.energy >= 64).all()) and bool((out.energy <= 192).all())
