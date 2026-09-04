"""Complex systems implemented in CAX."""

from .boids import Boids
from .elementary import Elementary
from .flow_lenia import FlowLenia
from .langton_ant import LangtonAnt
from .lenia import Lenia
from .life import Life
from .particle_lenia import ParticleLenia
from .particle_life import ParticleLife
from .reaction_diffusion import ReactionDiffusion
from .sandpile import Sandpile

__all__ = [
    "Boids",
    "Elementary",
    "FlowLenia",
    "LangtonAnt",
    "Lenia",
    "Life",
    "ParticleLenia",
    "ParticleLife",
    "ReactionDiffusion",
    "Sandpile",
]
