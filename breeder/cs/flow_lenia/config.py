"""Flow Lenia configuration."""

from functools import partial
from typing import Literal, override

from breeder.core import ComplexSystem
from breeder.cs.lenia import LeniaConfig, SampleConfig


class FlowLeniaConfig(LeniaConfig):
    """Configuration of the Flow Lenia complex system (arXiv:2212.07906).

    Flow Lenia is Lenia with a mass-conserving update: growth becomes a flow field and
    matter is transported by reintegration tracking instead of being created and
    destroyed. The genotype space is therefore *identical* to Lenia's — same rule
    parameters, same initial state — and this config subclasses `LeniaConfig`, adding
    only the flow parameters.

    Two consequences for the search, both intended:

    - **Mass is conserved exactly**, so the initial state fixes it forever and
    `min_mass`
        gates the seed rather than the dynamics. Localization (`min_concentration`) is
        the gate that does the work here.
    - The catalogued Lenia solitons are *not* Flow Lenia solitons (the update differs),
        so `sample.strategy: noise` — the paper's own random-patch seeding — is the
        principled start; `soliton_full` remains available as a Lenia-shaped seed.

    Attributes:
        name: Discriminator tag of the complex system.
        theta_A: Threshold of the flow activation (the paper's symbol); the official
        code
            uses `channel_size`, which `None` reproduces.
        n: Exponent of the flow activation.
        dd: Maximum displacement in pixels per step. Reintegration considers
            `(2·dd + 1)**num_spatial_dims` displacements per cell — the dominant cost.
        sigma: Spread of the displacement kernel.

    """

    name: Literal["flow_lenia"] = "flow_lenia"

    # The catalogued Lenia solitons are not Flow Lenia solitons: the paper's random
    # patch
    sample: SampleConfig = SampleConfig(strategy="noise")

    # The paper's own symbol for the flow-activation threshold, kept verbatim
    theta_A: float | None = None  # noqa: N815
    n: int = 2
    dd: int = 5
    sigma: float = 0.65

    @override
    def build(self) -> ComplexSystem:
        """Build the Flow Lenia complex system bound to this config."""
        # Deferred import: the sibling modules import FlowLeniaConfig at load time
        from breeder.cs.flow_lenia import SERIES, develop, mutate, sample, valid

        return ComplexSystem(
            sample_fn=partial(sample, config=self),
            mutate_fn=partial(mutate, config=self),
            develop_fn=partial(develop, config=self),
            valid_fn=partial(valid, config=self),
            spatial_dims=self.spatial_dims,
            unit=float(self.R * self.state_scale),
            series=SERIES,
            num_frames=self.num_steps,
        )
