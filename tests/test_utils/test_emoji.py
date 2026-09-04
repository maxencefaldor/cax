"""Tests for image utility functions."""

import io
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import PIL.Image
import pytest

from cax.utils.emoji import (
    get_emoji,
    get_emoji_array,
    get_emoji_filename,
    get_image_from_url,
)


@pytest.fixture
def mock_urlopen() -> Generator[MagicMock]:
    """Fixture to mock cax.utils.emoji.urlopen."""
    with patch("cax.utils.emoji.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"fake image content"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        yield mock_urlopen


@pytest.fixture
def mock_pil_image() -> Generator[MagicMock]:
    """Fixture to mock PIL.Image.open."""
    with patch("PIL.Image.open") as mock_open:
        mock_open.return_value = MagicMock(spec=PIL.Image.Image)
        yield mock_open


def test_get_image_from_url(mock_urlopen: MagicMock, mock_pil_image: MagicMock) -> None:
    """Test the get_image_from_url function."""
    url = "https://example.com/image.png"
    result = get_image_from_url(url)

    mock_urlopen.assert_called_once_with(url, timeout=30.0)

    # Check that PIL.Image.open was called with a BytesIO object containing the correct
    # content
    mock_pil_image.assert_called_once()
    call_arg = mock_pil_image.call_args[0][0]
    assert isinstance(call_arg, io.BytesIO)
    assert call_arg.getvalue() == b"fake image content"

    assert result == mock_pil_image.return_value


def test_get_image_from_url_raises_on_failure(mock_urlopen: MagicMock) -> None:
    """A download failure is reported as a ConnectionError, chaining the original."""
    from urllib.error import URLError

    mock_urlopen.side_effect = URLError("offline")

    with pytest.raises(ConnectionError, match="Could not download"):
        get_image_from_url("https://example.com/image.png")


@pytest.mark.parametrize(
    ("emoji", "filename"),
    [
        ("😀", "emoji_u1f600.png"),
        ("🐶", "emoji_u1f436.png"),
        # A sequence is spelled out in full, zero-width joiner included.
        ("👨‍💻", "emoji_u1f468_200d_1f4bb.png"),
        ("👍🏽", "emoji_u1f44d_1f3fd.png"),
        # The variation selector asking for emoji presentation is not part of the name.
        ("❤️", "emoji_u2764.png"),
        ("1️⃣", "emoji_u31_20e3.png"),
    ],
)
def test_get_emoji_filename(emoji: str, filename: str) -> None:
    """Codepoints are spelled in lowercase hexadecimal, joined by underscores."""
    assert get_emoji_filename(emoji) == filename


def test_get_emoji_filename_rejects_flags() -> None:
    """Noto keeps flags apart, named by country code, so they have no codepoint name."""
    with pytest.raises(ValueError, match="Flags"):
        get_emoji_filename("🇫🇷")


def test_get_emoji_filename_rejects_empty() -> None:
    """An empty string names nothing."""
    with pytest.raises(ValueError, match="empty"):
        get_emoji_filename("")


@pytest.mark.parametrize("emoji", ["😀", "🐶", "👨‍💻"])
def test_get_emoji(
    emoji: str, mock_urlopen: MagicMock, mock_pil_image: MagicMock
) -> None:
    """Test the get_emoji function."""
    result = get_emoji(emoji)

    url = mock_urlopen.call_args[0][0]
    assert url.endswith(f"/png/128/{get_emoji_filename(emoji)}")
    assert mock_urlopen.call_args[1] == {"timeout": 30.0}

    # Check that PIL.Image.open was called correctly (it's called inside
    # get_image_from_url)
    mock_pil_image.assert_called_once()
    call_arg = mock_pil_image.call_args[0][0]
    assert isinstance(call_arg, io.BytesIO)
    assert call_arg.getvalue() == b"fake image content"

    # Assert that the result is the mocked PIL image directly
    assert result == mock_pil_image.return_value


def test_get_emoji_pins_its_source(
    mock_urlopen: MagicMock, mock_pil_image: MagicMock
) -> None:
    """The glyphs come from a pinned revision, not from whatever a branch points at.

    An unpinned URL would make CAX depend on the current state of another project's
    default branch, where a rename would break every installed version at once.
    """
    get_emoji("🦖")

    url = mock_urlopen.call_args[0][0]
    assert "refs/heads/" not in url
    assert "@" in url, "the source revision should be pinned in the URL"


def test_get_emoji_array() -> None:
    """The array is resized, scaled to the unit interval, and framed in transparency."""
    glyph = PIL.Image.new("RGBA", (128, 128), (255, 128, 0, 255))

    with patch("cax.utils.emoji.get_emoji", return_value=glyph):
        array = get_emoji_array("🦎", size=8, pad_width=2)

    assert array.shape == (12, 12, 4)
    assert array.dtype == jnp.float32
    assert jnp.allclose(array[2:10, 2:10], jnp.array([1.0, 128 / 255, 0.0, 1.0]))

    # The frame is transparent, so a growing automaton is penalised for overshooting.
    assert jnp.all(array[:2] == 0.0)
    assert jnp.all(array[-2:] == 0.0)
    assert jnp.all(array[:, :2] == 0.0)
    assert jnp.all(array[:, -2:] == 0.0)


def test_get_emoji_array_without_padding() -> None:
    """Padding is optional, so the array is exactly the requested size."""
    glyph = PIL.Image.new("RGBA", (128, 128), (0, 0, 0, 255))

    with patch("cax.utils.emoji.get_emoji", return_value=glyph):
        array = get_emoji_array("🦎", size=16)

    assert array.shape == (16, 16, 4)
