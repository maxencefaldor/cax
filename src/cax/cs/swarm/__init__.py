"""Swarm 1-D tag consensus (isolated or chained workers)."""

from .cs import Swarm
from .perceive import SwarmPerceive
from .update import SwarmUpdate

__all__ = ["Swarm", "SwarmPerceive", "SwarmUpdate"]
