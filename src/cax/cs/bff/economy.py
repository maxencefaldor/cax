"""BFF economy module.

A BFF soup where copying costs something. Every program carries an energy count; each
epoch pays it an income and charges it one unit per step the pair spent with the
pointer in its half; a program whose energy is then below zero is dead, its bytes
replaced by random ones and its energy reset. Efficiency is now selected, a program
can make its partner pay, and there is a death other than being overwritten. This is
the resource that Tierra and Avida have and BFF lacks (see `notes/11`).

The soup dynamics are `BFF`'s, unchanged; this system wraps one.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem

from .cs import BFF, inverse_permutation


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EconomyState:
    """A soup and its programs' energy.

    Attributes:
        soup: Unsigned 8-bit array of shape (..., num_programs, tape_length).
        energy: Int32 array of shape (..., num_programs).

    """

    soup: Array
    energy: Array


class BFFEconomy(ComplexSystem[EconomyState, Array]):
    """A BFF soup with an energy economy; see the module docstring."""

    def __init__(self, *, income: int, bff: BFF):
        """Initialize the economy.

        Args:
            income: Energy paid to every program at every epoch.
            bff: The soup whose epochs are charged; its `rngs` also serve the deaths.

        """
        if income < 0:
            raise ValueError(f"income must be non-negative, got {income!r}")
        self.income = income
        self.bff = bff

    def init_state(
        self,
        *,
        num_programs: int | None = None,
        tape_length: int = 64,
        num_soups: int | None = None,
    ) -> EconomyState:
        """Create random soups whose programs all start with one income of energy.

        Args:
            num_programs: Number of programs; see `BFF.init_state`.
            tape_length: Bytes per program.
            num_soups: Number of independent soups, or None for one.

        Returns:
            An `EconomyState`.

        """
        soup = self.bff.init_state(
            num_programs=num_programs, tape_length=tape_length, num_soups=num_soups
        )
        energy = jnp.full(soup.shape[:-1], self.income, dtype=jnp.int32)
        return EconomyState(soup=soup, energy=energy)

    @override
    def _step(self, state: EconomyState, input: Array | None = None) -> EconomyState:
        *batch, num_programs, _ = state.soup.shape
        permutation = self.bff.pairing(num_programs, tuple(batch))
        soup, steps, first = self.bff.pair_and_run(state.soup, permutation)
        # Each program pays the steps spent in its half; the costs are in pair order
        # and come back to program order through the permutation's inverse.
        cost = jnp.stack([first, steps - first], axis=-1).reshape(permutation.shape)
        cost = jnp.take_along_axis(cost, inverse_permutation(permutation), axis=-1)
        energy = state.energy - cost + self.income
        dead = energy < 0
        replacement = jax.random.randint(
            self.bff.rngs.mutation(), soup.shape, 0, 256, dtype=jnp.uint8
        )
        soup = jnp.where(dead[..., None], replacement, soup)
        energy = jnp.where(dead, self.income, energy)
        return EconomyState(soup=soup, energy=energy)

    @nnx.jit
    @override
    def render(self, state: EconomyState) -> Array:
        """Render the soup as `BFF.render` does.

        Args:
            state: Economy state.

        Returns:
            RGB image with dtype uint8.

        """
        return self.bff.render(state.soup)
