"""Utilities for emojis."""

import io
from functools import cache
from urllib.error import URLError
from urllib.request import urlopen

import jax.numpy as jnp
import PIL.Image
from jax import Array
from PIL.Image import Image

_FETCH_TIMEOUT_S = 30.0

_NOTO_EMOJI_COMMIT = "8998f5dd683424a73e2314a8c1f1e359c19e8742"
"""The revision of Noto Emoji glyphs are fetched from.

Pinned rather than tracking a branch: an unpinned URL makes CAX's behaviour depend on
the current state of another project's default branch, so a rename there would break
every installed version of CAX at once. Bump this deliberately to pick up newly added
emoji.
"""

_EMOJI_PRESENTATION = "️"
"""Variation selector 16, which asks for the emoji rather than the text rendering.

Noto names its files by the base codepoints alone, so this is dropped when building one.
"""

_REGIONAL_INDICATORS = range(0x1F1E6, 0x1F200)
"""Codepoints that pair into flags, which Noto stores separately by country code."""


def get_image_from_url(url: str) -> Image:
    """Fetch an image from a given URL.

    Args:
        url: The URL of the image to fetch.

    Returns:
        The fetched image as a PIL Image object.

    Raises:
        ConnectionError: If the download fails — most commonly because the machine
            is offline. The original error is chained.

    """
    try:
        with urlopen(url, timeout=_FETCH_TIMEOUT_S) as response:
            image_data = response.read()
    except (URLError, TimeoutError) as error:
        raise ConnectionError(
            f"Could not download {url}. Emoji images are fetched from the network at "
            f"call time; check the connection and retry."
        ) from error

    image_pil = PIL.Image.open(io.BytesIO(image_data))
    return image_pil


def get_emoji_filename(emoji: str) -> str:
    """Build the Noto Emoji filename for an emoji.

    Noto names a glyph after the codepoints that spell it, in lowercase hexadecimal,
    joined by underscores. Sequences are spelled out in full, so the zero-width joiner
    of a glyph like 👨‍💻 is part of the name, while the variation selector that merely
    asks for an emoji presentation is not.

    Args:
        emoji: The emoji character or sequence.

    Returns:
        The filename, such as ``emoji_u1f468_200d_1f4bb.png``.

    Raises:
        ValueError: If ``emoji`` is empty, or is a flag. Noto keeps flags apart from the
            rest, named by country code rather than by codepoint.

    """
    codepoints = [
        ord(character) for character in emoji if character != _EMOJI_PRESENTATION
    ]
    if not codepoints:
        raise ValueError("Cannot build a filename for an empty emoji.")
    if any(codepoint in _REGIONAL_INDICATORS for codepoint in codepoints):
        raise ValueError(
            f"Flags such as {emoji!r} are not available: Noto Emoji stores them apart "
            f"from the other glyphs, named by country code rather than by codepoint."
        )

    return "emoji_u" + "_".join(f"{codepoint:x}" for codepoint in codepoints) + ".png"


@cache
def get_emoji(emoji: str) -> Image:
    """Fetch and return an emoji as a PIL Image.

    The glyph is downloaded from Google's Noto Emoji (PNG, 128 px) and cached in memory,
    so repeated calls for the same emoji fetch once. The image is returned without
    further processing; callers may convert to arrays or resize as needed.

    Args:
        emoji: The emoji character or sequence to fetch.

    Returns:
        A ``PIL.Image.Image`` instance containing the emoji.

    Raises:
        ValueError: If the emoji has no Noto glyph under this naming scheme.
        ConnectionError: If the download fails.

    """
    filename = get_emoji_filename(emoji)
    url = (
        f"https://cdn.jsdelivr.net/gh/googlefonts/noto-emoji@{_NOTO_EMOJI_COMMIT}"
        f"/png/128/{filename}"
    )
    return get_image_from_url(url)


def get_emoji_array(emoji: str, size: int, pad_width: int = 0) -> Array:
    """Fetch an emoji as a padded, premultiplied RGBA array.

    The glyph is resized to ``size`` and framed in transparent pixels, which is what a
    growing cellular automaton needs: the target sits in the middle of a larger grid, so
    the automaton has somewhere to overshoot into and can be penalised for doing so.

    Colour is premultiplied by alpha, CAX's convention for RGBA arrays: each pixel
    holds the light it emits, so a transparent pixel is zero in every channel and a
    loss on the array measures what is seen rather than the colour a PNG stores behind
    invisible pixels. ``rgba_to_rgb`` composites arrays in this convention.

    Args:
        emoji: The emoji character or sequence to fetch.
        size: Width and height, in pixels, to resize the glyph to.
        pad_width: Transparent pixels to add on each side.

    Returns:
        An array of shape ``(size + 2 * pad_width, size + 2 * pad_width, 4)`` holding
        premultiplied RGBA values in the unit interval.

    Raises:
        ValueError: If the emoji has no Noto glyph under this naming scheme.
        ConnectionError: If the download fails.

    """
    image_pil = get_emoji(emoji).resize(
        (size, size), resample=PIL.Image.Resampling.LANCZOS
    )
    array = jnp.asarray(image_pil, dtype=jnp.float32) / 255.0
    array = array.at[..., :3].multiply(array[..., 3:])
    return jnp.pad(array, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
