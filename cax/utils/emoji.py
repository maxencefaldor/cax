"""Utilities for emojis."""

import io
from urllib.request import urlopen

import jax
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


def get_emoji(emoji: str) -> jax.Array:
	"""Fetch, process, and return an emoji as a JAX array.

	Args:
		emoji: The emoji character to fetch.
		size: The desired size of the emoji image.
		padding: The amount of padding to add around the emoji.

	Returns:
		The processed emoji image.

	"""
	# Get the emoji image
	code = hex(ord(emoji))[2:].lower()
	url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
	image_pil = get_image_from_url(url)
	return image_pil
