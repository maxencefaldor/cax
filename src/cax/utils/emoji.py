"""Utilities for emojis."""

import io
from functools import cache
from urllib.error import URLError
from urllib.request import urlopen

import PIL.Image
from PIL.Image import Image

_FETCH_TIMEOUT_S = 30.0


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


@cache
def get_emoji(emoji: str) -> Image:
	"""Fetch and return an emoji as a PIL Image.

	The emoji glyph is downloaded from Google's Noto Emoji repository (PNG, 128 px) and
	cached in memory, so repeated calls for the same glyph fetch once. The image is
	returned as a PIL Image without further processing. Callers may convert to arrays or
	resize as needed.

	Args:
		emoji: The emoji character to fetch.

	Returns:
		A ``PIL.Image.Image`` instance containing the emoji.

	"""
	# Get the emoji image
	code = hex(ord(emoji))[2:].lower()
	url = f"https://raw.githubusercontent.com/googlefonts/noto-emoji/refs/heads/main/png/128/emoji_u{code}.png"
	image_pil = get_image_from_url(url)
	return image_pil
