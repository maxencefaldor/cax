"""Utility functions for image processing."""

import io

import jax
import jax.numpy as jnp
import PIL.Image
import requests
from PIL.Image import Image


def get_image_from_url(url: str) -> Image:
	"""Fetch an image from a given URL.

	Args:
		url: The URL of the image to fetch.

	Returns:
		The fetched image as a PIL Image object.

	"""
	r = requests.get(url)
	image_pil = PIL.Image.open(io.BytesIO(r.content))
	return image_pil


def get_emoji(emoji: str, *, size: int, padding: int) -> jax.Array:
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

	# Resize and pad the image
	image_pil.thumbnail((size, size), resample=PIL.Image.Resampling.LANCZOS)
	image = jnp.array(image_pil, dtype=jnp.float32) / 255.0
	image = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)))

	# Multiply the RGB values by the alpha channel
	image = image.at[..., :3].set(image[..., :3] * image[..., 3:])
	return image
