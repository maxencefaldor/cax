"""Tests for image utility functions."""

import io
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest

from cax.utils.emoji import get_emoji, get_image_from_url


@pytest.fixture
def mock_urlopen() -> Generator[MagicMock]:
	"""Fixture to mock cax.utils.emoji.urlopen."""
	with patch("cax.utils.emoji.urlopen") as mock_urlopen:
		yield mock_urlopen


@pytest.fixture
def mock_pil_image() -> Generator[MagicMock]:
	"""Fixture to mock PIL.Image.open."""
	with patch("PIL.Image.open") as mock_open:
		yield mock_open


def test_get_image_from_url(mock_urlopen: MagicMock, mock_pil_image: MagicMock) -> None:
	"""Test the get_image_from_url function."""
	mock_response = MagicMock()
	mock_response.read.return_value = b"fake image content"
	mock_context_manager = MagicMock()
	mock_context_manager.__enter__.return_value = mock_response
	mock_urlopen.return_value = mock_context_manager

	mock_image = MagicMock(spec=PIL.Image.Image)
	mock_pil_image.return_value = mock_image

	url = "https://example.com/image.png"
	result = get_image_from_url(url)

	mock_urlopen.assert_called_once_with(url)

	# Check that PIL.Image.open was called with a BytesIO object containing the correct content
	mock_pil_image.assert_called_once()
	call_arg = mock_pil_image.call_args[0][0]
	assert isinstance(call_arg, io.BytesIO)
	assert call_arg.getvalue() == b"fake image content"

	assert result == mock_image


@pytest.mark.parametrize("emoji", ["😀", "🐶"])
def test_get_emoji(emoji: str, mock_urlopen: MagicMock, mock_pil_image: MagicMock) -> None:
	"""Test the get_emoji function."""
	mock_response = MagicMock()
	mock_response.read.return_value = b"fake image content"
	mock_context_manager = MagicMock()
	mock_context_manager.__enter__.return_value = mock_response
	mock_urlopen.return_value = mock_context_manager

	mock_image = MagicMock(spec=PIL.Image.Image)
	mock_pil_image.return_value = mock_image

	result = get_emoji(emoji)

	expected_url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{hex(ord(emoji))[2:].lower()}.png?raw=true"
	mock_urlopen.assert_called_once_with(expected_url)

	# Check that PIL.Image.open was called correctly (it's called inside get_image_from_url)
	mock_pil_image.assert_called_once()
	call_arg = mock_pil_image.call_args[0][0]
	assert isinstance(call_arg, io.BytesIO)
	assert call_arg.getvalue() == b"fake image content"

	# Assert that the result is the mocked PIL image directly
	assert result == mock_image
