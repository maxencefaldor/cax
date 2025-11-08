"""Perception modules for complex systems.

These modules gather neighborhood information from the state to produce a perception that
downstream updates consume. Implementations include neighborhood-based sampling and convolutional
variants.
"""

from .conv_perceive import ConvPerceive
from .kernels import grad2_kernel, grad_kernel, identity_kernel, neighbors_kernel
from .moore_perceive import MoorePerceive
from .perceive import Perceive, Perception
from .von_neumann_perceive import VonNeumannPerceive

__all__ = [
	"Perceive",
	"Perception",
	"MoorePerceive",
	"VonNeumannPerceive",
	"ConvPerceive",
	"identity_kernel",
	"neighbors_kernel",
	"grad_kernel",
	"grad2_kernel",
]
