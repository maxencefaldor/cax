"""Rendering utilities."""

import jax.numpy as jnp


def clip_and_uint8(frame):
	"""Clip values to valid range and convert to uint8."""
	frame = jnp.clip(frame, min=0.0, max=1.0)
	return (frame * 255).astype(jnp.uint8)


def hsv_to_rgb(hsv):
	"""Convert HSV to RGB."""
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


def rgb_to_hsv(rgb):
	"""Convert RGB to HSV."""
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
