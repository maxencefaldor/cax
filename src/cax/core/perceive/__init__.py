"""Perception modules for Cellular Automata.

This module contains various perception mechanisms used in Cellular Automata
implementations. These perception modules are responsible for gathering
information from the neighborhood of each cell, which is then used to
determine the cell's next state.

These modules are designed to work with the CAX library, providing flexible
and efficient perception options for different types of cellular automata
models.
"""

from .conv_perceive import ConvPerceive
from .kernels import grad2_kernel, grad_kernel, identity_kernel, neighbors_kernel
from .moore_perceive import MoorePerceive
from .perceive import Perceive
from .von_neumann_perceive import VonNeumannPerceive

__all__ = [
	"ConvPerceive",
	"MoorePerceive",
	"Perceive",
	"VonNeumannPerceive",
	"identity_kernel",
	"neighbors_kernel",
	"grad_kernel",
	"grad2_kernel",
]
