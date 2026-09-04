"""Life module.

References:
    [1] Mathematical Games: The fantastic combinations of John Conway's new solitaire
        game "life", Martin Gardner. 1970.

"""

from .cs import Life
from .perceive import LifePerceive
from .update import LifeUpdate

__all__ = [
    "Life",
    "LifePerceive",
    "LifeUpdate",
]
