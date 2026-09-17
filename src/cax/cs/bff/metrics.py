"""BFF soup metrics module.

Host-side measurements of a soup of tapes. Compression is not jittable, so these take
NumPy arrays and are meant to run on saved states, not inside a scan.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108

"""

import zlib

import numpy as np


def byte_entropy(soup: np.ndarray) -> float:
    """Shannon entropy of the byte distribution of a soup, in bits per byte.

    Args:
        soup: Unsigned 8-bit array of any shape.

    Returns:
        Entropy in bits per byte, between 0 and 8.

    """
    counts = np.bincount(np.asarray(soup, dtype=np.uint8).ravel(), minlength=256)
    frequency = counts[counts > 0] / counts.sum()
    return float(-(frequency * np.log2(frequency)).sum())


def compressed_bits_per_byte(soup: np.ndarray, *, compressor: str = "brotli") -> float:
    """Compressed size of a soup in bits per byte, the Kolmogorov complexity proxy.

    The paper compresses the whole soup with brotli at quality 2 and a 24-bit window.
    zlib is offered as a dependency-free fallback; its absolute values differ.

    Args:
        soup: Unsigned 8-bit array of any shape; compressed in row-major order.
        compressor: `"brotli"` (needs the `brotli` package) or `"zlib"`.

    Returns:
        Compressed size in bits divided by the number of bytes.

    """
    data = np.ascontiguousarray(np.asarray(soup, dtype=np.uint8)).tobytes()
    if compressor == "brotli":
        import brotli

        size = len(brotli.compress(data, mode=brotli.MODE_GENERIC, quality=2, lgwin=24))
    elif compressor == "zlib":
        size = len(zlib.compress(data, level=6))
    else:
        raise ValueError(f"Unknown compressor {compressor!r}")
    return 8.0 * size / len(data)


def high_order_entropy(soup: np.ndarray, *, compressor: str = "brotli") -> float:
    """High-order entropy of a soup: byte entropy minus compressed bits per byte.

    Near zero for a uniformly random soup and for a soup of one repeated byte; large
    when the soup is made of long repeated strings, which is what a soup taken over by
    a self-replicator looks like. The paper calls a run transitioned when this reaches 1
    (Figure 6); the reference scripts use 3.

    Args:
        soup: Unsigned 8-bit array of any shape.
        compressor: `"brotli"` or `"zlib"`; see `compressed_bits_per_byte`.

    Returns:
        High-order entropy in bits per byte.

    """
    return byte_entropy(soup) - compressed_bits_per_byte(soup, compressor=compressor)
