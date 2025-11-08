"""Utilities for emojis."""

import io
from urllib.request import urlopen

import PIL.Image
from PIL.Image import Image


def get_image_from_url(url: str) -> Image:
	"""Fetch an image from a given URL.

	Args:
		url: The URL of the image to fetch.

	Returns:
		The fetched image as a PIL Image object.

	"""
	with urlopen(url) as response:
		image_data = response.read()

	image_pil = PIL.Image.open(io.BytesIO(image_data))
	return image_pil


def get_emoji(emoji: str) -> Image:
	"""Fetch and return an emoji as a PIL Image.

	The emoji glyph is downloaded from Google's Noto Emoji repository (PNG, 128 px). The image
	is returned as a PIL Image without further processing. Callers may convert to arrays or
	resize as needed.

	Args:
		emoji: The emoji character to fetch.

	Returns:
		A ``PIL.Image.Image`` instance containing the emoji.

	"""
	# Get the emoji image
	code = hex(ord(emoji))[2:].lower()
	url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
	image_pil = get_image_from_url(url)
	return image_pil
