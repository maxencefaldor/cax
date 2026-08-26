"""Boids module.

References:
	[1] Flocks, Herds, and Schools: A Distributed Behavioral Model, Craig W. Reynolds.
		1987.

"""

from .cs import Boids
from .perceive import BoidsPerceive
from .perception import BoidsPerception
from .policy import BoidsPolicy
from .state import BoidsState
from .update import BoidsUpdate

__all__ = [
	"Boids",
	"BoidsPerceive",
	"BoidsPerception",
	"BoidsPolicy",
	"BoidsState",
	"BoidsUpdate",
]
