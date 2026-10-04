"""Tests for MQRI — including payload rebuild (the D-gap)."""

from __future__ import annotations

import numpy as np
import pytest

from cax.cs.mqri.plate import (
    decode_payload,
    encode_payload,
    freeze_mask,
    reflection_id,
    strip_rows_for,
    tamper,
)

DEMO = b"SLID-PHI|unit-pack|demo-2026-08-17"


def _numpy_step(state: np.ndarray, freeze: np.ndarray) -> np.ndarray:
    alive = state[..., 0]
    phase = state[..., 1]
    neigh = np.zeros_like(alive)
    phase_sum = np.zeros_like(phase)
    for di, dj in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        a = np.roll(alive, (di, dj), axis=(-2, -1))
        p = np.roll(phase, (di, dj), axis=(-2, -1))
        neigh = neigh + a
        phase_sum = phase_sum + p * a
    new_alive = (
        ((neigh >= 2) & (neigh <= 4)) | ((alive >= 1) & (neigh >= 1))
    ).astype(np.float32)
    safe_n = np.maximum(neigh, 1)
    avg = np.where(neigh > 0, np.floor(phase_sum / safe_n) % 4, phase)
    new_phase = np.where(new_alive >= 1, avg, 0).astype(np.float32)
    nxt = np.stack([new_alive, new_phase], axis=-1)
    return np.where(freeze > 0, state, nxt)


def test_payload_roundtrip_after_rollout() -> None:
    """Decode must rebuild the payload from levels alone after N steps."""
    rows = strip_rows_for(len(DEMO), 32)
    seed = encode_payload(DEMO, height=32, width=32)
    freeze = freeze_mask(32, 32, rows)
    state = seed.copy()
    for _ in range(12):
        state = _numpy_step(state, freeze)
    got = decode_payload(state, len(DEMO), strip_rows=rows)
    assert got == DEMO


def test_identity_stable_and_tamper_changes() -> None:
    """Re-read ID matches; flipping free cells must change it."""
    rows = strip_rows_for(len(DEMO), 32)
    seed = encode_payload(DEMO, height=32, width=32)
    freeze = freeze_mask(32, 32, rows)
    state = seed.copy()
    for _ in range(12):
        state = _numpy_step(state, freeze)
    ident = reflection_id(state)
    assert reflection_id(state) == ident
    damaged = tamper(state, n_cells=8, strip_rows=rows)
    assert reflection_id(damaged) != ident


def test_distinct_payloads_distinct_ids() -> None:
    """Two payloads must not share an identity."""
    a = encode_payload(DEMO, height=32, width=32)
    b = encode_payload(b"different-payload-xyz", height=32, width=32)
    assert reflection_id(a) != reflection_id(b)


def test_fiducial_survives_rollout() -> None:
    """The 3x3 corner stays alive with phase 1."""
    rows = strip_rows_for(len(DEMO), 32)
    seed = encode_payload(DEMO, height=32, width=32)
    freeze = freeze_mask(32, 32, rows)
    state = seed.copy()
    for _ in range(12):
        state = _numpy_step(state, freeze)
    assert np.all(state[:3, :3, 0] >= 0.5)
    assert np.allclose(state[:3, :3, 1], 1.0)


def test_mqri_jit_init() -> None:
    """Test that Mqri can be instantiated under jax.jit when jaxlib works."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    try:
        jax.numpy.zeros(1)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from cax.cs.mqri import Mqri

    @jax.jit
    def init_mqri() -> Mqri:
        return Mqri(height=16, width=16, strip_rows=1)

    init_mqri()
