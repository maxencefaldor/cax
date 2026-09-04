"""Flow Lenia development module.

Development instantiates a CAX `FlowLenia` from the genotype's rule parameters and
observes it exactly as Lenia is observed — same simulation, same toroidal metric series,
same centered rendering — because the two systems differ only in the update rule.
"""

from cax.cs.flow_lenia import FlowLenia

from ...core.phenotype import Phenotype
from ..lenia import Genotype, observe
from .config import FlowLeniaConfig


def develop(
    genotype: Genotype, config: FlowLeniaConfig, *, center: bool = True
) -> Phenotype:
    """Develop a genotype into its phenotype.

    Args:
        genotype: Genotype to develop.
        config: Flow Lenia configuration.
        center: Whether frames are centered on the pattern. Descriptors want the
        centered
            view (translation invariance); visualizations want the raw view, which
            conveys motion.

    Returns:
        The phenotype.

    """
    cs = FlowLenia(
        spatial_dims=config.spatial_dims,
        channel_size=config.channel_size,
        R=config.R,
        T=config.T,
        state_scale=config.state_scale,
        rule_params=genotype.rule_params,
        theta_A=config.theta_A,
        n=config.n,
        dd=config.dd,
        sigma=config.sigma,
    )
    return observe(cs, genotype.state_init, config, center=center)
