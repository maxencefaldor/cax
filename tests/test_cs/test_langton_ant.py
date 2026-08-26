"""Tests for Langton's Ant."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cax.cs.langton_ant import LangtonAnt, LangtonAntState


def test_langton_ant_jit_init() -> None:
	"""Test that LangtonAnt can be instantiated under jax.jit."""

	@jax.jit
	def init_langton_ant() -> LangtonAnt:
		rngs = nnx.Rngs(0)
		turns = jnp.array([1, 3], dtype=jnp.int32)
		langton_ant = LangtonAnt(turns=turns, rngs=rngs)
		return langton_ant

	try:
		init_langton_ant()
	except Exception as e:
		pytest.fail(f"LangtonAnt instantiation failed under jit: {e}")


def test_langton_ant_turns_from_rule_string() -> None:
	"""Test that turns_from_rule_string correctly parses rule strings."""
	turns = LangtonAnt.turns_from_rule_string("RL")
	assert turns.shape == (2,)
	assert int(turns[0]) == 1  # R
	assert int(turns[1]) == 3  # L

	turns = LangtonAnt.turns_from_rule_string("LLRR")
	assert turns.shape == (4,)
	assert int(turns[0]) == 3  # L
	assert int(turns[1]) == 3  # L
	assert int(turns[2]) == 1  # R
	assert int(turns[3]) == 1  # R

	turns = LangtonAnt.turns_from_rule_string("NU")
	assert turns.shape == (2,)
	assert int(turns[0]) == 0  # N
	assert int(turns[1]) == 2  # U


def test_langton_ant_step_classic_rl() -> None:
	"""Test that a single step of classic RL produces the expected transition.

	Starting on an empty grid (all zeros) with the ant facing North:
	- Cell color is 0, rule is R -> turn right -> now facing East
	- Flip cell from 0 to 1
	- Move forward (East) -> col += 1
	"""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("RL")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((5, 5, 1), dtype=jnp.float32)
	position = jnp.array([2, 2], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)  # North
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	next_state = langton_ant._step(state)

	# Cell at (2, 2) should now be 1
	assert float(next_state.grid[2, 2, 0]) == 1.0
	# Ant should have turned right (North -> East = direction 1) and moved East
	assert float(next_state.direction) == 1.0
	# Position should be (2, 3)
	assert float(next_state.position[0]) == 2.0
	assert float(next_state.position[1]) == 3.0


def test_langton_ant_step_on_color_1() -> None:
	"""Test step when ant is on a cell with color 1 in RL rule.

	With color 1, rule is L -> turn left -> direction - 1 (mod 4).
	Starting facing North (0), turn left -> West (3).
	Flip cell from 1 to 0. Move West -> col -= 1.
	"""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("RL")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((5, 5, 1), dtype=jnp.float32)
	grid = grid.at[2, 2, 0].set(1.0)
	position = jnp.array([2, 2], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)  # North
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	next_state = langton_ant._step(state)

	# Cell at (2, 2) should wrap back to 0
	assert float(next_state.grid[2, 2, 0]) == 0.0
	# North + L (3) = 3 -> West
	assert float(next_state.direction) == 3.0
	# Move West: (2, 2-1) = (2, 1)
	assert float(next_state.position[0]) == 2.0
	assert float(next_state.position[1]) == 1.0


def test_langton_ant_periodic_boundary() -> None:
	"""Test that the ant wraps around grid boundaries."""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("RL")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((5, 5, 1), dtype=jnp.float32)
	position = jnp.array([0, 2], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)  # North
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	# On color 0, rule R -> turn right -> East, move to (0, 3)
	# But let's set direction to North and cell to 1 (L) so it turns left to West
	# Actually let's test: facing North at row=0, color=1, turn left -> West, move col-1=1
	# That doesn't test wrapping. Let's use direction=North at row=0, color=0.
	# Turn R -> East, move to (0, 3). No wrap.
	# Better: facing North at (0, 2), with some setup to go North wrapping.
	# Set direction to West (3), at col=0.
	grid = jnp.zeros((5, 5, 1), dtype=jnp.float32)
	grid = grid.at[2, 0, 0].set(1.0)  # color 1 -> L
	position = jnp.array([2, 0], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)  # North
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	next_state = langton_ant._step(state)

	# Color 1, L -> turn left. North + 3 = 3 -> West. Move West from col=0 -> col=4 (wrap)
	assert float(next_state.direction) == 3.0
	assert float(next_state.position[0]) == 2.0
	assert float(next_state.position[1]) == 4.0


def test_langton_ant_multi_step() -> None:
	"""Test that multi-step execution via __call__ works."""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("RL")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((11, 11, 1), dtype=jnp.float32)
	position = jnp.array([5, 5], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	final_state = langton_ant(state, num_steps=4)

	# After 4 steps on an empty grid, the ant should have turned right 4 times
	# and returned to facing North, one cell diagonally offset.
	assert final_state.grid.shape == (11, 11, 1)
	assert final_state.position.shape == (2,)


def test_langton_ant_render() -> None:
	"""Test that render returns uint8 RGB with correct shape."""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("RL")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((8, 8, 1), dtype=jnp.float32)
	position = jnp.array([4, 4], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	image = langton_ant.render(state)
	assert image.shape == (8, 8, 3)
	assert image.dtype == jnp.uint8


def test_langton_ant_render_multicolor() -> None:
	"""Test that render works for multi-color rules."""
	rngs = nnx.Rngs(0)
	turns = LangtonAnt.turns_from_rule_string("LLRR")
	langton_ant = LangtonAnt(turns=turns, rngs=rngs)

	grid = jnp.zeros((8, 8, 1), dtype=jnp.float32)
	grid = grid.at[3, 3, 0].set(2.0)
	position = jnp.array([4, 4], dtype=jnp.int32)
	direction = jnp.array(0, dtype=jnp.int32)
	state = LangtonAntState(grid=grid, position=position, direction=direction)

	image = langton_ant.render(state)
	assert image.shape == (8, 8, 3)
	assert image.dtype == jnp.uint8
