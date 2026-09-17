"""BFF assay module.

Controlled experiments on single programs, so that a soup can be judged on more than
whether it copies. Each assay places a program in a pair with chosen partners, runs the
pair, and reads one number off the result. Together they give a program a behaviour
vector: does it copy alone, only among its own kind, only on a host; does it survive
being run over; does it overwrite or reproduce its partner. A lineage landing where no
lineage sat before is what higher-order emergence looks like in these numbers. The
mutational scan gives the same program's core (the bytes it cannot lose) and its
robustness.

All assays are batched over programs and jitted; on a GPU they run through the kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .detector import NUM_CHAINS, replication_score
from .interpreter import Control, run


def run_pairs(
    first: Array,
    second: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    control: Control = "matched",
) -> Array:
    """Run each program of `first` paired with the matching program of `second`.

    Args:
        first: Unsigned 8-bit array of shape (num, length), the first halves.
        second: Unsigned 8-bit array of shape (num, length), the second halves.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per pair.
        control: Control-flow rule.

    Returns:
        Unsigned 8-bit array of shape (num, 2 * length), the pairs after execution.

    """
    tapes = jnp.concatenate([first, second], axis=-1)
    return run(tapes, opcode_table, num_steps=num_steps, control=control)[0]


@partial(jax.jit, static_argnames=("num_steps", "control"))
def assays(
    programs: Array,
    random_partners: Array,
    soup_partners: Array,
    host: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    control: Control = "matched",
) -> dict[str, Array]:
    """The behaviour vector of each program.

    Args:
        programs: Unsigned 8-bit array of shape (num, length).
        random_partners: Unsigned 8-bit array of shape (num, NUM_CHAINS, length) of
            uniformly random tapes; see `sample_partners`.
        soup_partners: Same shape, tapes drawn from the soup the programs live in.
        host: Unsigned 8-bit array of shape (length,), the soup's dominant program.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per execution.
        control: Control-flow rule.

    Returns:
        A dict of float32 arrays of shape (num,), each in [0, 1]:

        - `replicates`: detector score against random partners, over the length.
        - `replicates_in_soup`: the same against partners from the soup; much higher
          than `replicates` means the program needs a host, a parasite.
        - `replicates_with_kin`: the same with copies of itself as partners.
        - `survives`: as the second half of a pair with a random first half, the
          fraction of its bytes intact afterwards.
        - `survives_in_soup`: the same with first halves from the soup.
        - `overwrites_host`: as the first half of a pair with the soup's dominant
          program as second half, the fraction of the host's bytes changed.
        - `copies_partner`: as the first half with a random partner, the fraction of
          its own bytes that afterwards equal the partner's: reproducing the other.

    """
    num, length = programs.shape
    kin = jnp.broadcast_to(programs[:, None, :], random_partners.shape)
    score = partial(
        replication_score,
        opcode_table=opcode_table,
        num_steps=num_steps,
        control=control,
    )
    execute = partial(
        run_pairs, opcode_table=opcode_table, num_steps=num_steps, control=control
    )

    def survival(first: Array) -> Array:
        # (num, NUM_CHAINS, length) first halves against the program as second half.
        flat = first.reshape(-1, length)
        second = jnp.repeat(programs, NUM_CHAINS, axis=0)
        out = execute(flat, second)[:, length:].reshape(num, NUM_CHAINS, length)
        return jnp.mean(out == programs[:, None, :], axis=(1, 2))

    hosts = jnp.broadcast_to(host, programs.shape)
    after_host = execute(programs, hosts)[:, length:]
    partner = random_partners[:, 0, :]
    after_partner = execute(programs, partner)[:, :length]
    return {
        "replicates": score(programs, random_partners) / length,
        "replicates_in_soup": score(programs, soup_partners) / length,
        "replicates_with_kin": score(programs, kin) / length,
        "survives": survival(random_partners),
        "survives_in_soup": survival(soup_partners),
        "overwrites_host": jnp.mean(after_host != hosts, axis=-1),
        "copies_partner": jnp.mean(after_partner == partner, axis=-1),
    }


@partial(jax.jit, static_argnames=("num_steps", "control", "variants", "threshold"))
def mutational_scan(
    program: Array,
    key: Array,
    opcode_table: Array,
    *,
    num_steps: int = 8192,
    control: Control = "matched",
    variants: int = 4,
    threshold: int = 48,
) -> Array:
    """How often a program still replicates when one byte is changed, per position.

    Args:
        program: Unsigned 8-bit array of shape (length,).
        key: PRNG key for the replacement bytes and the partners.
        opcode_table: Integer array of shape (256,) mapping bytes to `Op` values.
        num_steps: Step budget per execution.
        control: Control-flow rule.
        variants: Replacement values tried per position.
        threshold: Detector score at which a mutant still counts as replicating; the
            2026 paper's 48 suits programs that copy a whole 64-byte tape.

    Returns:
        Float32 array of shape (length,): the fraction of `variants` single-byte
            mutants at each position whose detector score stays at or above
            `threshold`. Positions near zero are the program's core; the mean is its
            robustness.

    """
    length = program.shape[0]
    key_byte, key_partner = jax.random.split(key)
    replacement = jax.random.randint(
        key_byte, (length, variants), 1, 256, dtype=jnp.uint8
    )
    replacement = (program[:, None] + replacement).astype(
        jnp.uint8
    )  # never the original
    position = jnp.arange(length)
    mutants = jnp.where(
        position[:, None, None] == position[None, None, :],
        replacement[:, :, None],
        program[None, None, :],
    ).reshape(length * variants, length)
    partners = jax.random.randint(
        key_partner, (length * variants, NUM_CHAINS, length), 0, 256, dtype=jnp.uint8
    )
    scores = replication_score(
        mutants, partners, opcode_table, num_steps=num_steps, control=control
    )
    return jnp.mean((scores >= threshold).reshape(length, variants), axis=-1)
