"""Utilities for rendering."""

import jax.numpy as jnp
from jax import Array


def rgba_to_rgb(array: Array) -> Array:
	"""Convert an RGBA image to RGB by alpha compositing over white.

	The function assumes the last dimension encodes channels and that the input is normalized
	to the range ``[0, 1]`` with shape ``(..., 4)``. The output preserves the input shape
	except for the channel dimension, which becomes ``3``.

	Args:
		array: RGBA image with shape ``(..., 4)`` and values in ``[0, 1]``.

	Returns:
		RGB image with shape ``(..., 3)`` and values in ``[0, 1]``.

	"""
	if array.shape[-1] != 4:
		raise ValueError(f"Expected an RGBA array with 4 channels, got {array.shape[-1]}")
	rgb, alpha = array[..., :-1], array[..., -1:]
	alpha = jnp.clip(alpha, min=0.0, max=1.0)
	return (1.0 - alpha) * 1.0 + alpha * rgb


def rgb_to_hsv(rgb: Array) -> Array:
	"""Convert RGB to HSV.

	Input and output are in the range ``[0, 1]`` and use channel-last layout.

	Args:
		rgb: RGB image with shape ``(..., 3)``.

	Returns:
		HSV image with shape ``(..., 3)``.

	"""
	input_shape = rgb.shape
	rgb = rgb.reshape(-1, 3)
	r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

	maxc = jnp.maximum(jnp.maximum(r, g), b)
	minc = jnp.minimum(jnp.minimum(r, g), b)
	v = maxc
	deltac = maxc - minc

	s = jnp.where(maxc != 0, deltac / maxc, 0)

	deltac = jnp.where(deltac == 0, 1, deltac)  # Avoid division by zero

	rc = (maxc - r) / deltac
	gc = (maxc - g) / deltac
	bc = (maxc - b) / deltac

	h = jnp.where(r == maxc, bc - gc, jnp.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))

	h = jnp.where(minc == maxc, 0.0, h)
	h = (h / 6.0) % 1.0

	hsv = jnp.stack([h, s, v], axis=-1)
	return hsv.reshape(input_shape)


def hsv_to_rgb(hsv: Array) -> Array:
	"""Convert HSV to RGB.

	Input and output are in the range ``[0, 1]`` and use channel-last layout.

	Args:
		hsv: HSV image with shape ``(..., 3)``.

	Returns:
		RGB image with shape ``(..., 3)``.

	"""
	input_shape = hsv.shape
	hsv = hsv.reshape(-1, 3)
	h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

	i = jnp.floor(h * 6.0).astype(jnp.int32)
	f = (h * 6.0) - i
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))

	i = i % 6

	rgb = jnp.zeros_like(hsv)
	rgb = jnp.where((i == 0)[..., None], jnp.stack([v, t, p], axis=-1), rgb)
	rgb = jnp.where((i == 1)[..., None], jnp.stack([q, v, p], axis=-1), rgb)
	rgb = jnp.where((i == 2)[..., None], jnp.stack([p, v, t], axis=-1), rgb)
	rgb = jnp.where((i == 3)[..., None], jnp.stack([p, q, v], axis=-1), rgb)
	rgb = jnp.where((i == 4)[..., None], jnp.stack([t, p, v], axis=-1), rgb)
	rgb = jnp.where((i == 5)[..., None], jnp.stack([v, p, q], axis=-1), rgb)

	rgb = jnp.where(s[..., None] == 0.0, jnp.full_like(rgb, v[..., None]), rgb)

	return rgb.reshape(input_shape)


def clip_and_uint8(frame: Array) -> Array:
	"""Clip a floating-point image to ``[0, 1]`` and convert to ``uint8``.

	Args:
		frame: Image-like array with values expected in or near ``[0, 1]``.

	Returns:
		Array of dtype ``uint8`` with values in ``[0, 255]``.

	"""
	frame = jnp.clip(frame, min=0.0, max=1.0)
	return (frame * 255).astype(jnp.uint8)


def render_array_with_channels_to_rgb(array: Array) -> Array:
	"""Render an array with channels as an RGB image.

	This function processes an input array and converts it into an RGB image based on the number of
	channels present in the array. The conversion logic is as follows:
	- If the array has 1 channel, it is repeated across the RGB channels to produce a grayscale
		image.
	- If the array has 2 channels, the first channel is interpreted as hue and the second as
		saturation. These are converted to RGB using a fixed brightness value, resulting in a
		colorful representation.
	- If the array has 3 or more channels, the last three channels are used directly as the RGB
		values.

	Args:
		array: Input array with shape ``(..., C)`` and values in ``[0, 1]``.

	Returns:
		RGB array with shape ``(..., 3)`` and values in ``[0, 1]``.

	"""
	num_channels = array.shape[-1]

	if num_channels == 1:
		# 1 channel
		rgb = jnp.repeat(array, 3, axis=-1)
	elif num_channels == 2:
		# 2 channels
		hue = array[..., 0:1]  # Use the first channel as hue
		saturation = array[..., 1:2]  # and the second as saturation
		value = jnp.ones_like(hue)  # Use full brightness
		hsv = jnp.concatenate([hue, saturation, value], axis=-1)
		rgb = hsv_to_rgb(hsv)
	else:
		# 3 channels or more
		rgb = array[..., -3:]

	return rgb


def render_array_with_channels_to_rgba(array: Array) -> Array:
	"""Render an array with channels as an RGBA image.

	This function processes an input array and converts it into an RGBA image based on the number of
	channels present in the array. The conversion logic is as follows:
	- If the array has 1 channel, it is repeated across the RGBA channels.
	- If the array has 2 channels, the first channel is used for RGB, and the second for alpha.
	- If the array has 3 channels, the first channel is interpreted as hue and the second as
		saturation. These are converted to RGB using a fixed brightness value, and the last channel
		is used as the alpha channel.
	- If the array has 4 or more channels, the last four channels are used directly as RGBA.

	Args:
		array: Input array with shape ``(..., C)`` and values in ``[0, 1]``.

	Returns:
		RGBA array with shape ``(..., 4)`` and values in ``[0, 1]``.

	"""
	num_channels = array.shape[-1]

	if num_channels == 1:
		# 1 channel
		rgba = jnp.repeat(array, 4, axis=-1)
	elif num_channels == 2:
		# 2 channels
		rgb = jnp.repeat(array[..., 0:1], 3, axis=-1)
		alpha = array[..., 1:2]
		rgba = jnp.concatenate([rgb, alpha], axis=-1)
	elif num_channels == 3:
		# 3 channels
		hue = array[..., 0:1]  # Use the first channel as hue
		saturation = array[..., 1:2]  # and the second as saturation
		value = jnp.ones_like(hue)  # Use full brightness
		hsv = jnp.concatenate([hue, saturation, value], axis=-1)
		rgb = hsv_to_rgb(hsv)
		alpha = array[..., 2:3]  # Use the last channel as alpha
		rgba = jnp.concatenate([rgb, alpha], axis=-1)
	else:
		# 4 or more channels
		rgba = array[..., -4:]

	return rgba


def pixel_grid(resolution: int, *, low: float = 0.0, high: float = 1.0) -> Array:
	"""Build a square grid of pixel-center coordinates.

	Args:
		resolution: Number of pixels along each side.
		low: Coordinate of the first pixel along each axis.
		high: Coordinate of the last pixel along each axis.

	Returns:
		Array with shape ``(resolution, resolution, 2)`` of ``(x, y)`` coordinates.

	"""
	x = jnp.linspace(low, high, resolution)
	y = jnp.linspace(low, high, resolution)
	return jnp.stack(jnp.meshgrid(x, y), axis=-1)


def nearest_point(grid: Array, points: Array) -> tuple[Array, Array]:
	"""Find the nearest of `points` for every grid pixel.

	Args:
		grid: Pixel coordinates with shape ``(resolution, resolution, 2)``.
		points: Point coordinates with shape ``(num_points, 2)``.

	Returns:
		A ``(min_distance_sq, index)`` tuple of ``(resolution, resolution)`` arrays:
			the squared distance to, and the index of, the nearest point per pixel.

	"""
	distance_sq = jnp.sum((grid[:, :, None, :] - points[None, None, :, :]) ** 2, axis=-1)
	return jnp.min(distance_sq, axis=-1), jnp.argmin(distance_sq, axis=-1)


def soft_disk_mask(min_distance_sq: Array, radius: float) -> Array:
	"""Anti-aliased disk coverage from squared distances to the nearest point.

	Args:
		min_distance_sq: Squared distance to the nearest point per pixel.
		radius: Disk radius in the grid's coordinate space.

	Returns:
		Coverage in ``[0, 1]``: one at the point, falling to zero at the disk edge.

	"""
	return jnp.clip(1.0 - min_distance_sq / (radius**2), 0.0, 1.0)
