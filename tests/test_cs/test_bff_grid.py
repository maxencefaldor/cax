"""Tests for the BFF grid machine."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cax.cs.bff import opcode_table_from_string, parse, run
from cax.cs.bff.grid import BFFGrid, Control, GridState, Turn

TABLE = opcode_table_from_string()


def enriched_tape(key: jax.Array, length: int = 128) -> jax.Array:
    """A tape with half instruction bytes, so brackets and copies are frequent."""
    key_coin, key_op, key_byte = jax.random.split(key, 3)
    ops = jnp.asarray(parse("[]+-.,<>{}"))
    coin = jax.random.bernoulli(key_coin, 0.5, (length,))
    op = ops[jax.random.randint(key_op, (length,), 0, ops.shape[0])]
    byte = jax.random.randint(key_byte, (length,), 0, 256).astype(jnp.uint8)
    return jnp.where(coin, op, byte)


def single_thread(memory: jax.Array, position: tuple[int, ...], direction: int):
    """A state with one thread at `position` heading `direction`, heads on it."""
    p = jnp.asarray([position], dtype=jnp.int32)
    return GridState(
        memory=memory,
        position=p,
        direction=jnp.asarray([direction], dtype=jnp.int32),
        head0=p,
        head1=p,
        origin=p,
        age=jnp.zeros((1,), dtype=jnp.int32),
        step=jnp.zeros((), dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    ("control", "turn"),
    [("flip", "quarter"), ("flip", "reflect"), ("cyclic", "reflect")],
)
def test_grid_1d_matches_interpreter(control: Control, turn: Turn) -> None:
    cs = BFFGrid(
        num_steps=10**6, mutation_rate=0.0, control=control, turn=turn, rngs=nnx.Rngs(0)
    )
    for seed in range(6):
        tape = enriched_tape(jax.random.key(seed))
        state = cs(single_thread(tape, (0,), 0), num_steps=400)
        expected, _, _ = run(
            tape[None], TABLE, num_steps=400, control=control, implementation="xla"
        )
        np.testing.assert_array_equal(np.asarray(state.memory), np.asarray(expected[0]))


def test_grid_cyclic_jumps_along_a_column() -> None:
    memory = jnp.zeros((6, 4), dtype=jnp.uint8)
    memory = memory.at[1, 2].set(ord("[")).at[4, 2].set(ord("]"))
    state = single_thread(memory, (1, 2), 0)  # heading +axis 0
    state = GridState(**{**vars(state), "head0": jnp.asarray([[0, 0]], jnp.int32)})
    cs = BFFGrid(mutation_rate=0.0, control="cyclic", rngs=nnx.Rngs(0))
    out = cs(state, num_steps=1)
    assert out.position[0].tolist() == [5, 2]
    assert int(out.direction[0]) == 0


def test_grid_2d_step_and_render() -> None:
    cs = BFFGrid(rngs=nnx.Rngs(0))
    state = cs.init_state(shape=(16, 32))
    assert state.position.shape == (8, 2)
    state = cs(state, num_steps=3)
    assert state.memory.shape == (16, 32)
    assert int(state.step) == 3
    assert bool((state.position >= 0).all()) and bool((state.position[:, 1] < 32).all())
    rgb = cs.render(state)
    assert rgb.shape == (16, 32, 3) and rgb.dtype == jnp.uint8
    assert cs.render(cs.init_state(shape=(64,), num_threads=2)).shape == (1, 64, 3)
    assert cs.render(cs.init_state(shape=(4, 8, 8))).shape == (8, 8, 3)


@pytest.mark.parametrize(
    ("turn", "direction", "position"), [("quarter", 2, [7, 0]), ("reflect", 3, [0, 7])]
)
def test_grid_close_bracket_turns(
    turn: Turn, direction: int, position: list[int]
) -> None:
    memory = jnp.zeros((8, 8), dtype=jnp.uint8).at[0, 0].set(ord("]")).at[3, 3].set(1)
    state = single_thread(memory, (0, 0), 1)  # heading +axis 1
    state = GridState(**{**vars(state), "head0": jnp.asarray([[3, 3]], jnp.int32)})
    cs = BFFGrid(mutation_rate=0.0, turn=turn, rngs=nnx.Rngs(0))
    out = cs(state, num_steps=1)
    assert int(out.direction[0]) == direction
    assert out.position[0].tolist() == position


def test_grid_conflict_picks_one_writer() -> None:
    memory = jnp.zeros((16,), dtype=jnp.uint8).at[0].set(ord(".")).at[1].set(ord("+"))
    memory = memory.at[5].set(7)
    # Thread 0 copies cell 5 (7) into cell 9; thread 1 increments cell 9.
    state = GridState(
        memory=memory,
        position=jnp.asarray([[0], [1]], jnp.int32),
        direction=jnp.zeros((2,), jnp.int32),
        head0=jnp.asarray([[5], [9]], jnp.int32),
        head1=jnp.asarray([[9], [1]], jnp.int32),
        origin=jnp.asarray([[0], [1]], jnp.int32),
        age=jnp.zeros((2,), jnp.int32),
        step=jnp.zeros((), jnp.int32),
    )
    cs = BFFGrid(mutation_rate=0.0, rngs=nnx.Rngs(0))
    outcomes = {int(cs(state, num_steps=1).memory[9]) for _ in range(6)}
    assert outcomes <= {7, 1}


def test_grid_respawn_resets_heads_and_age() -> None:
    cs = BFFGrid(num_steps=4, mutation_rate=0.0, rngs=nnx.Rngs(0))
    state = cs.init_state(shape=(8, 8), num_threads=16)
    state = GridState(**{**vars(state), "age": jnp.full((16,), 3, jnp.int32)})
    out = cs(state, num_steps=1)
    assert bool((out.age == 0).all())
    assert bool((out.head0 == out.position).all()) and bool(
        (out.head1 == out.position).all()
    )


def test_grid_window_confines_threads() -> None:
    cs = BFFGrid(num_steps=10**6, mutation_rate=0.0, window=8, rngs=nnx.Rngs(0))
    state = cs.init_state(shape=(64, 64), num_threads=32)
    out = cs(state, num_steps=200)
    for field in (out.position, out.head0, out.head1):
        offset = (field - out.origin + 32) % 64 - 32
        assert bool((offset >= -4).all()) and bool((offset < 4).all())


def test_grid_anchor_spawns_on_the_lattice() -> None:
    cs = BFFGrid(num_steps=4, mutation_rate=0.0, window=16, anchor=8, rngs=nnx.Rngs(0))
    state = cs.init_state(shape=(64, 64), num_threads=64)

    def on_lattice(state: GridState) -> bool:
        axis = state.direction % 2
        start = jnp.take_along_axis(state.position, axis[:, None], axis=1)[:, 0]
        return bool((start % 8 == 0).all())

    assert on_lattice(state)
    ahead = (state.origin - state.position) % 64
    assert bool(((ahead == 8) | (ahead == 56)).sum(axis=-1).all())
    state = GridState(**{**vars(state), "age": jnp.full((64,), 3, jnp.int32)})
    out = cs(state, num_steps=1)
    assert bool((out.age == 0).all()) and on_lattice(out)


def test_grid_positive_headings() -> None:
    cs = BFFGrid(num_steps=2, mutation_rate=0.0, headings="positive", rngs=nnx.Rngs(0))
    state = cs.init_state(shape=(16, 16, 16), num_threads=128)
    assert bool((state.direction < 3).all())
    out = cs(
        GridState(**{**vars(state), "age": jnp.ones((128,), jnp.int32)}), num_steps=1
    )
    assert bool((out.direction < 3).all())


def test_grid_relative_heads_run_the_mirror_backwards() -> None:
    tape = enriched_tape(jax.random.key(3), 64)
    memory = jnp.zeros((256,), jnp.uint8).at[64:128].set(tape)
    forward = BFFGrid(
        num_steps=10**6,
        mutation_rate=0.0,
        control="cyclic",
        window=128,
        rngs=nnx.Rngs(0),
    )
    p = jnp.asarray([[64]], jnp.int32)
    state = GridState(
        memory=memory,
        position=p,
        direction=jnp.zeros((1,), jnp.int32),
        head0=p,
        head1=p,
        origin=jnp.asarray([[128]], jnp.int32),
        age=jnp.zeros((1,), jnp.int32),
        step=jnp.zeros((), jnp.int32),
    )
    out = np.asarray(forward(state, num_steps=500).memory)
    # The same tape laid backwards at 128..192, run by a backward thread from 191.
    mirrored = jnp.zeros((256,), jnp.uint8).at[128:192].set(tape[::-1])
    q = jnp.asarray([[191]], jnp.int32)
    state = GridState(
        memory=mirrored,
        position=q,
        direction=jnp.ones((1,), jnp.int32),
        head0=q,
        head1=q,
        origin=jnp.asarray([[128]], jnp.int32),
        age=jnp.zeros((1,), jnp.int32),
        step=jnp.zeros((), jnp.int32),
    )
    backward = BFFGrid(
        num_steps=10**6,
        mutation_rate=0.0,
        control="cyclic",
        window=128,
        heads="relative",
        rngs=nnx.Rngs(0),
    )
    out_back = np.asarray(backward(state, num_steps=500).memory)
    np.testing.assert_array_equal(out_back[64:192][::-1], out[64:192])
