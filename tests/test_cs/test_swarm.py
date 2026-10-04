"""Tests for 1-D swarm tag consensus."""

from __future__ import annotations

import numpy as np
import pytest

IDLE, LIT, RUN, MATCH = 0.0, 1.0, 2.0, 3.0


def _step(tags: np.ndarray, *, worker_size: int, isolated: bool) -> np.ndarray:
    left = np.roll(tags, 1)
    right = np.roll(tags, -1)
    if isolated:
        idx = np.arange(tags.size)
        left = np.where((idx % worker_size) == 0, IDLE, left)
        right = np.where((idx % worker_size) == worker_size - 1, IDLE, right)

    def structured(a: np.ndarray) -> np.ndarray:
        return (np.isclose(a, RUN)) | (np.isclose(a, MATCH))

    pair_lr = np.isclose(left, right) & structured(left)
    pair_lm = np.isclose(left, tags) & structured(left)
    pair_rm = np.isclose(right, tags) & structured(right)
    nxt = np.where(pair_lr, left, tags)
    nxt = np.where(pair_lm, left, nxt)
    nxt = np.where(pair_rm, right, nxt)
    return nxt


def test_run_spreads_inside_a_worker() -> None:
    """Two run neighbors pull a lit cell to run."""
    tags = np.array([RUN, LIT, RUN], dtype=np.float32)
    out = _step(tags, worker_size=3, isolated=False)
    assert np.isclose(out[1], RUN)


def test_isolated_does_not_cross_the_cut() -> None:
    """A run on both sides of the cut must not pull the last cell of worker 0."""
    worker = 4
    tags = np.zeros(8, dtype=np.float32)
    tags[worker - 2] = RUN
    tags[worker - 1] = LIT
    tags[worker] = RUN
    iso = _step(tags, worker_size=worker, isolated=True)
    chained = _step(tags, worker_size=worker, isolated=False)
    assert np.isclose(iso[worker - 1], LIT)
    assert np.isclose(chained[worker - 1], RUN)


def test_match_pair_beats_lit() -> None:
    """Left and self match keep match; a lone match does not overwrite idle."""
    tags = np.array([MATCH, MATCH, LIT], dtype=np.float32)
    out = _step(tags, worker_size=8, isolated=False)
    assert np.isclose(out[0], MATCH)
    assert np.isclose(out[1], MATCH)


def test_swarm_jit_init() -> None:
    """Instantiate Swarm under jax.jit when jaxlib works."""
    jax = pytest.importorskip("jax")
    try:
        jax.numpy.zeros(1)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from cax.cs.swarm import Swarm

    @jax.jit
    def init_swarm() -> Swarm:
        return Swarm(worker_size=8, isolated=True)

    init_swarm()
