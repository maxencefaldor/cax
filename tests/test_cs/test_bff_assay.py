"""Tests for the BFF assays."""

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import opcode_table_from_string, parse, sample_partners
from cax.cs.bff.assay import assays, mutational_scan, run_pairs

TABLE = opcode_table_from_string()
REPLICATOR = "[[{.>]-]A]-]>.{[["


def program(text: str) -> jnp.ndarray:
    """Parse a program into a zero-padded 64-byte tape."""
    bytes_ = parse(text)
    return jnp.zeros((64,), jnp.uint8).at[: bytes_.shape[0]].set(bytes_)


def test_assays_of_a_replicator_and_of_noise() -> None:
    key = jax.random.key(0)
    noise = jax.random.randint(jax.random.key(1), (64,), 0, 256, dtype=jnp.uint8)
    programs = jnp.stack([program(REPLICATOR), noise])
    random_partners = sample_partners(key, 2)
    soup_partners = sample_partners(jax.random.key(2), 2)
    out = assays(
        programs,
        random_partners,
        soup_partners,
        program(REPLICATOR),
        TABLE,
        num_steps=8192,
    )
    assert set(out) == {
        "replicates",
        "replicates_in_soup",
        "replicates_with_kin",
        "survives",
        "survives_in_soup",
        "overwrites_host",
        "copies_partner",
        "spends",
        "partner_pays",
    }
    for value in out.values():
        assert value.shape == (2,)
        assert bool(((value >= 0) & (value <= 1)).all())
    # The hand-written replicator copies its 17 bytes, so it scores 17 of 64.
    assert float(out["replicates"][0]) > 0.25
    assert float(out["replicates"][1]) < 0.1
    assert float(out["replicates_with_kin"][0]) > 0.25
    assert float(out["overwrites_host"][0]) > float(out["overwrites_host"][1])
    # The replicator loops to the budget; noise runs off its half in a few steps.
    assert float(out["spends"][0]) + float(out["partner_pays"][0]) > 0.9
    assert float(out["spends"][1]) < 0.1


def test_run_pairs_shape() -> None:
    first = jax.random.randint(jax.random.key(0), (3, 64), 0, 256, dtype=jnp.uint8)
    second = jax.random.randint(jax.random.key(1), (3, 64), 0, 256, dtype=jnp.uint8)
    out = run_pairs(first, second, TABLE, num_steps=64)
    assert out.shape == (3, 128) and out.dtype == jnp.uint8


def test_mutational_scan_finds_the_core() -> None:
    scan = np.asarray(
        mutational_scan(
            program(REPLICATOR),
            jax.random.key(0),
            TABLE,
            num_steps=8192,
            variants=2,
            threshold=12,
        )
    )
    assert scan.shape == (64,)
    # The padding bytes are free to change; the program's own bytes mostly are not.
    assert scan[len(REPLICATOR) :].mean() > 0.9
    assert scan[: len(REPLICATOR)].mean() < 0.7
