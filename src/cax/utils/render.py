"""Utilities for rendering."""

from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive.kernels import HEX_BASIS


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


def _resample(array: Array, transform: Array) -> Array:
	"""Resample the two spatial axes through a change of basis, wrapping at the edges.

	Leading axes are treated as batch, so a stacked trajectory resamples like a single
	image does.
	"""
	height, width = array.shape[-3], array.shape[-2]
	rows, columns = jnp.meshgrid(
		jnp.arange(height, dtype=jnp.float32),
		jnp.arange(width, dtype=jnp.float32),
		indexing="ij",
	)
	index = jnp.stack([rows, columns], axis=-1) @ transform
	coordinates = [index[..., 0] % height, index[..., 1] % width]

	def sample(image: Array) -> Array:
		return jnp.stack(
			[
				jax.scipy.ndimage.map_coordinates(image[..., c], coordinates, order=1, mode="wrap")
				for c in range(image.shape[-1])
			],
			axis=-1,
		)

	flat = array.reshape(-1, height, width, array.shape[-1])
	return jax.vmap(sample)(flat).reshape(array.shape)


def hex_to_square(array: Array) -> Array:
	"""Resample a triangular-lattice array onto a square pixel grid.

	A triangular lattice is stored in an ordinary square array whose axes stand for the
	lattice vectors `(1, 0)` and `(1/2, sqrt(3)/2)` rather than for a Cartesian frame.
	Drawn directly such an array leans over, because the viewer reads its axes as
	perpendicular when they are sixty degrees apart. This maps each output pixel back
	through the basis and samples there, so what is drawn is the lattice as it actually
	sits in the plane.

	The lattice is treated as periodic, matching the wrap-around a cellular automaton on a
	torus already assumes.

	Args:
		array: Values on a triangular lattice, with shape `(..., height, width, channels)`.
			Leading axes are treated as batch.

	Returns:
		An array of the same shape, holding the lattice resampled onto square pixels.

	"""
	return _resample(array, jnp.linalg.inv(HEX_BASIS))


def square_to_hex(array: Array) -> Array:
	"""Resample a square-pixel array onto a triangular lattice.

	The inverse of `hex_to_square`, and what a picture needs before it is placed on a
	triangular lattice: written in directly it would be sheared, and a shape that is no
	longer itself is no longer sustained by a rule that was tuned to it.

	Args:
		array: Values on square pixels, with shape `(..., height, width, channels)`.
			Leading axes are treated as batch.

	Returns:
		An array of the same shape, holding the picture resampled onto the lattice.

	"""
	return _resample(array, HEX_BASIS)


class _Renderable(Protocol):
	"""A complex system that can draw one state."""

	def render(self, state: Any, **kwargs: Any) -> Array: ...


def render_states(cs: _Renderable, states: Any, *, batch_size: int = 16, **kwargs: Any) -> Array:
	"""Render every state of a trajectory, a batch of frames at a time.

	Rendering a particle system costs one `(resolution^2, num_particles)` array per frame,
	which is far larger than the frame it produces. Vectorizing over a whole trajectory
	asks for all of them at once — terabytes for a long run — and survives only where the
	compiler happens to fuse the intermediate away, so the same notebook runs on an
	accelerator and dies on a CPU. Rendering in batches bounds the peak at `batch_size`
	frames whatever the backend, and costs nothing: the frames are independent.

	Args:
		cs: The complex system, whose `render` draws one state.
		states: A trajectory: a pytree whose leaves have the time steps on axis 0.
		batch_size: Frames rendered at once. Larger is faster and needs more memory.
		**kwargs: Forwarded to `cs.render` (`resolution`, `particle_radius`, ...).

	Returns:
		The rendered frames, with shape `(num_steps, resolution, resolution, 3)`.

	"""
	num_steps = jax.tree.leaves(states)[0].shape[0]
	render = nnx.vmap(lambda cs, state: cs.render(state, **kwargs), in_axes=(None, 0))

	def batch(start: int) -> Array:
		return render(cs, jax.tree.map(lambda leaf: leaf[start : start + batch_size], states))

	return jnp.concatenate([batch(start) for start in range(0, num_steps, batch_size)])
