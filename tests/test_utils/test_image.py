"""Tests for image utility functions."""

import io
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import PIL.Image
import pytest
from cax.utils.image import get_emoji, get_image_from_url


@pytest.fixture
def mock_requests_get() -> Generator[MagicMock, None, None]:
	"""Fixture to mock requests.get."""
	with patch("requests.get") as mock_get:
		yield mock_get


@pytest.fixture
def mock_pil_image() -> Generator[MagicMock, None, None]:
	"""Fixture to mock PIL.Image.open."""
	with patch("PIL.Image.open") as mock_open:
		yield mock_open


def test_get_image_from_url(mock_requests_get: MagicMock, mock_pil_image: MagicMock) -> None:
	"""Test the get_image_from_url function."""
	mock_response = MagicMock()
	mock_response.content = b"fake image content"
	mock_requests_get.return_value = mock_response
	mock_image = MagicMock(spec=PIL.Image.Image)
	mock_pil_image.return_value = mock_image

	url = "https://example.com/image.png"
	result = get_image_from_url(url)

	mock_requests_get.assert_called_once_with(url)

	# Check that PIL.Image.open was called with a BytesIO object containing the correct content
	mock_pil_image.assert_called_once()
	call_arg = mock_pil_image.call_args[0][0]
	assert isinstance(call_arg, io.BytesIO)
	assert call_arg.getvalue() == b"fake image content"

	assert result == mock_image


@pytest.mark.parametrize(
	"emoji, size, padding",
	[
		("😀", 32, 2),
		("🐶", 64, 4),
	],
)
def test_get_emoji(
	emoji: str, size: int, padding: int, mock_requests_get: MagicMock, mock_pil_image: MagicMock
) -> None:
	"""Test the get_emoji function."""
	mock_response = MagicMock()
	mock_response.content = b"fake image content"
	mock_requests_get.return_value = mock_response
	mock_image = MagicMock(spec=PIL.Image.Image)
	mock_image.size = (size, size)
	mock_pil_image.return_value = mock_image

	with patch("jax.numpy.array") as mock_jnp_array:
		mock_jnp_array.return_value = jnp.ones((size, size, 4))
		result = get_emoji(emoji, size=size, padding=padding)

	expected_url = (
		f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{hex(ord(emoji))[2:].lower()}.png?raw=true"
	)
	mock_requests_get.assert_called_once_with(expected_url)

	# Update this assertion
	mock_pil_image.assert_called_once()
	call_arg = mock_pil_image.call_args[0][0]
	assert isinstance(call_arg, io.BytesIO)
	assert call_arg.getvalue() == b"fake image content"

	mock_image.thumbnail.assert_called_once_with((size, size), resample=PIL.Image.Resampling.LANCZOS)

	assert result.shape == (size + 2 * padding, size + 2 * padding, 4)
