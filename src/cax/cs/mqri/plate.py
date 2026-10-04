"""MQRI plate helpers: payload strip, decode, and reflection identity.

The local NCA is not invertible. Decode (the D-gap) plants payload bits in a
frozen bottom strip and reads them back after rollout. The 3x3 fiducial is
frozen the same way so geometry stays recoverable. Identity hashes synthetic
Stokes-like I/Q/U of the quantized plate; it is not a substitute for decode.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
from numpy.typing import NDArray

N_PHASE = 4
FIDUCIAL = 3
BITS_PER_CELL = 2
PROBE_TAG = b"MQRI-v0"


def strip_rows_for(nbytes: int, width: int) -> int:
    """Number of bottom rows needed to store ``nbytes`` at 2 bits per cell."""
    cells = (nbytes * 8 + BITS_PER_CELL - 1) // BITS_PER_CELL
    rows = (cells + width - 1) // width
    return max(1, rows)


def freeze_mask(height: int, width: int, strip_rows: int) -> NDArray[np.float32]:
    """Fiducial corner plus payload strip. Shape ``(height, width, 1)``."""
    if strip_rows + FIDUCIAL > height:
        raise ValueError(
            f"strip_rows={strip_rows} plus fiducial={FIDUCIAL} exceeds height={height}"
        )
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[:FIDUCIAL, :FIDUCIAL, 0] = 1.0
    mask[height - strip_rows :, :, 0] = 1.0
    return mask


def encode_payload(
    payload: bytes, *, height: int = 32, width: int = 32
) -> NDArray[np.float32]:
    """Write payload bits into the frozen strip of a fresh plate.

    Args:
        payload: Bytes to store. Must fit in the bottom strip at 2 bits/cell.
        height: Grid height.
        width: Grid width.

    Returns:
        State array ``(height, width, 2)`` with alive in channel 0 and phase
        (0..3) in channel 1.

    """
    rows = strip_rows_for(len(payload), width)
    mask = freeze_mask(height, width, rows)
    state = np.zeros((height, width, 2), dtype=np.float32)
    state[:FIDUCIAL, :FIDUCIAL, 0] = 1.0
    state[:FIDUCIAL, :FIDUCIAL, 1] = 1.0

    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    # Pad to a multiple of BITS_PER_CELL.
    pad = (-len(bits)) % BITS_PER_CELL
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    phases = bits.reshape(-1, BITS_PER_CELL)
    values = phases[:, 0] * 2 + phases[:, 1]

    strip = np.argwhere(mask[:, :, 0] > 0)
    # Payload cells are the strip only, not the fiducial.
    payload_cells = strip[strip[:, 0] >= height - rows]
    if values.size > len(payload_cells):
        raise ValueError(
            f"payload of {len(payload)} B needs {values.size} cells, "
            f"strip has {len(payload_cells)}"
        )
    for k, val in enumerate(values):
        i, j = payload_cells[k]
        state[i, j, 0] = 1.0
        state[i, j, 1] = float(val)
    return state


def decode_payload(
    state: NDArray[np.floating], nbytes: int, *, strip_rows: int | None = None
) -> bytes:
    """Rebuild payload bytes from the frozen strip. This is the D-gap."""
    height, width = int(state.shape[-3]), int(state.shape[-2])
    rows = strip_rows if strip_rows is not None else strip_rows_for(nbytes, width)
    mask = freeze_mask(height, width, rows)
    payload_cells = np.argwhere(
        (mask[:, :, 0] > 0) & (np.arange(height)[:, None] >= height - rows)
    )
    n_bits = nbytes * 8
    n_cells = (n_bits + BITS_PER_CELL - 1) // BITS_PER_CELL
    bits: list[int] = []
    for k in range(n_cells):
        i, j = payload_cells[k]
        val = int(np.round(state[i, j, 1])) & 3
        bits.append((val >> 1) & 1)
        bits.append(val & 1)
    packed = np.packbits(np.array(bits[:n_bits], dtype=np.uint8))
    return packed.tobytes()[:nbytes]


def levels(state: NDArray[np.floating]) -> NDArray[np.int8]:
    """Map alive/phase to optical levels: 0 empty, 1..4 phase bins."""
    alive = state[..., 0]
    phase = np.clip(np.round(state[..., 1]), 0, N_PHASE - 1)
    return np.where(alive >= 0.5, phase + 1, 0).astype(np.int8)


def _soft_blur(x: NDArray[np.floating]) -> NDArray[np.floating]:
    p = np.pad(x, 1, mode="edge")
    return (
        p[1:-1, 1:-1] * 0.4
        + p[0:-2, 1:-1] * 0.15
        + p[2:, 1:-1] * 0.15
        + p[1:-1, 0:-2] * 0.15
        + p[1:-1, 2:] * 0.15
    )


def reflection_id(state: NDArray[np.floating], probe_tag: bytes = PROBE_TAG) -> str:
    """Stable identity from synthetic I/Q/U of the quantized plate."""
    lev = levels(state).astype(np.float64)
    intensity = (lev > 0).astype(np.float64) * (0.4 + 0.15 * lev)
    ang = np.where(lev > 0, (lev - 1) * (np.pi / N_PHASE), 0.0)
    stokes = {
        "I": _soft_blur(intensity),
        "Q": _soft_blur(intensity * np.cos(2 * ang)),
        "U": _soft_blur(intensity * np.sin(2 * ang)),
    }
    parts = [probe_tag]
    for key in ("I", "Q", "U"):
        a = stokes[key]
        h, w = a.shape
        bh, bw = h // 8, w // 8
        grid = []
        for i in range(8):
            for j in range(8):
                block = a[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                grid.append(round(float(block.mean()), 4))
        parts.append(struct.pack(f"{len(grid)}d", *grid))
    return hashlib.sha256(b"".join(parts)).hexdigest()[:32]


def tamper(
    state: NDArray[np.floating],
    n_cells: int = 8,
    *,
    strip_rows: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Flip ``n_cells`` free (unfrozen) cells so identity must change."""
    rng = rng or np.random.default_rng(0)
    out = np.array(state, copy=True)
    height, width = out.shape[0], out.shape[1]
    rows = strip_rows if strip_rows is not None else 1
    freeze = freeze_mask(height, width, rows)[:, :, 0] > 0
    free_i, free_j = np.where(~freeze)
    if free_i.size == 0:
        return out
    pick = rng.choice(free_i.size, size=min(n_cells, free_i.size), replace=False)
    for idx in pick:
        i, j = int(free_i[idx]), int(free_j[idx])
        out[i, j, 0] = 1.0
        out[i, j, 1] = float(rng.integers(0, N_PHASE))
    return out
