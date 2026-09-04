"""Particle Life module.

References:
    [1] Particle Life, Tom Mohr. 2018. https://particle-life.com/

"""

from .cs import ParticleLife
from .perceive import ParticleLifePerceive
from .perception import ParticleLifePerception
from .state import ParticleLifeState
from .update import ParticleLifeUpdate

__all__ = [
    "ParticleLife",
    "ParticleLifePerceive",
    "ParticleLifePerception",
    "ParticleLifeState",
    "ParticleLifeUpdate",
]
