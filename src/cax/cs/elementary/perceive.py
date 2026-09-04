"""Elementary Cellular Automata perceive module.

This module implements the perception function for Elementary Cellular Automata. It
extracts the three-cell neighborhood (self, right, left) for each cell using the Moore
neighborhood.
"""

from typing import Literal

from cax.core.perceive import MoorePerceive


class ElementaryPerceive(MoorePerceive):
    """Elementary Cellular Automata perception.

    Extracts the three-cell neighborhood for each cell in a one-dimensional cellular
    automaton. The perception consists of three channels ordered as [self, right, left]
    corresponding to the Moore neighborhood output for 1D radius 1.
    """

    def __init__(self, *, padding: Literal["CIRCULAR", "ZERO"] = "CIRCULAR"):
        """Initialize Elementary perceive.

        Args:
            padding: Boundary condition mode. "CIRCULAR" for periodic boundaries,
                "ZERO" for a border of permanently dead cells.

        """
        super().__init__(num_spatial_dims=1, radius=1, padding=padding)
