"""MQRI plate: frozen-strip payload plus reflection identity.

The local rule is a public toy (majority life and phase diffusion), not a
trained NCA. Decode rebuilds bytes from the frozen strip after rollout.
"""

from .cs import Mqri
from .perceive import MqriPerceive
from .plate import (
    decode_payload,
    encode_payload,
    reflection_id,
    strip_rows_for,
    tamper,
)
from .update import MqriUpdate

__all__ = [
    "Mqri",
    "MqriPerceive",
    "MqriUpdate",
    "decode_payload",
    "encode_payload",
    "reflection_id",
    "strip_rows_for",
    "tamper",
]
