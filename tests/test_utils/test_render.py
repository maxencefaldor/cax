"""Tests for rendering utility functions."""

import jax.numpy as jnp
import pytest
from jax import Array

from cax.utils.render import (
	clip_and_uint8,
	hsv_to_rgb,
	render_array_with_channels_to_rgb,
	render_array_with_channels_to_rgba,
	rgb_to_hsv,
	rgba_to_rgb,
)


@pytest.mark.parametrize(
	"rgba_input, expected_rgb",
	[
		# Simple case: Fully opaque red
		(jnp.array([1.0, 0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0])),
		# Simple case: Fully transparent blue (should be white background)
		(jnp.array([0.0, 0.0, 1.0, 0.0]), jnp.array([1.0, 1.0, 1.0])),
		# Simple case: 50% transparent green
		(jnp.array([0.0, 1.0, 0.0, 0.5]), jnp.array([0.5, 1.0, 0.5])),
		# Multi-pixel case
		(
			jnp.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),
			jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
		),
		# Multi-dimensional case (e.g., image)
		(
			jnp.array([[[1.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.5]]]),
			jnp.array([[[1.0, 0.0, 0.0]], [[0.5, 1.0, 0.5]]]),
		),
		# Alpha clipping
		(jnp.array([0.5, 0.5, 0.5, 1.5]), jnp.array([0.5, 0.5, 0.5])),  # Alpha > 1
		(jnp.array([0.5, 0.5, 0.5, -0.5]), jnp.array([1.0, 1.0, 1.0])),  # Alpha < 0 -> white
	],
)
def test_rgba_to_rgb(rgba_input: Array, expected_rgb: Array) -> None:
	"""Test the rgba_to_rgb function."""
	result = rgba_to_rgb(rgba_input)
	assert result.shape == expected_rgb.shape
	assert jnp.allclose(result, expected_rgb)


@pytest.mark.parametrize(
	"rgb_input, expected_hsv",
	[
		# Basic colors
		(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 1.0])),  # Red
		(jnp.array([0.0, 1.0, 0.0]), jnp.array([1 / 3, 1.0, 1.0])),  # Green
		(jnp.array([0.0, 0.0, 1.0]), jnp.array([2 / 3, 1.0, 1.0])),  # Blue
		(jnp.array([1.0, 1.0, 0.0]), jnp.array([1 / 6, 1.0, 1.0])),  # Yellow
		(jnp.array([0.0, 1.0, 1.0]), jnp.array([0.5, 1.0, 1.0])),  # Cyan
		(jnp.array([1.0, 0.0, 1.0]), jnp.array([5 / 6, 1.0, 1.0])),  # Magenta
		# Grayscale
		(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0])),  # Black
		(jnp.array([1.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 1.0])),  # White
		(jnp.array([0.5, 0.5, 0.5]), jnp.array([0.0, 0.0, 0.5])),  # Gray
		# Multi-pixel
		(
			jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
			jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
		),
		# Multi-dimensional
		(
			jnp.array([[[1.0, 1.0, 1.0]], [[0.0, 1.0, 0.0]]]),
			jnp.array([[[0.0, 0.0, 1.0]], [[1 / 3, 1.0, 1.0]]]),
		),
	],
)
def test_rgb_hsv_conversion(rgb_input: Array, expected_hsv: Array) -> None:
	"""Test the rgb_to_hsv and hsv_to_rgb conversions."""
	# Test RGB to HSV
	hsv_result = rgb_to_hsv(rgb_input)
	assert hsv_result.shape == expected_hsv.shape
	# Need tolerance for floating point comparisons
	assert jnp.allclose(hsv_result, expected_hsv, atol=1e-6)

	# Test HSV back to RGB
	rgb_result = hsv_to_rgb(expected_hsv)
	assert rgb_result.shape == rgb_input.shape
	assert jnp.allclose(rgb_result, rgb_input, atol=1e-6)


@pytest.mark.parametrize(
	"input_frame, expected_output",
	[
		# Values within range
		(jnp.array([0.0, 0.5, 1.0]), jnp.array([0, 127, 255], dtype=jnp.uint8)),
		# Values outside range (clipping)
		(jnp.array([-0.5, 1.5, 0.2]), jnp.array([0, 255, 51], dtype=jnp.uint8)),
		# Multi-dimensional
		(
			jnp.array([[0.1, 1.1], [-0.1, 0.9]]),
			jnp.array([[25, 255], [0, 229]], dtype=jnp.uint8),
		),
	],
)
def test_clip_and_uint8(input_frame: Array, expected_output: Array) -> None:
	"""Test the clip_and_uint8 function."""
	result = clip_and_uint8(input_frame)
	print(result)
	assert result.dtype == jnp.uint8
	assert result.shape == expected_output.shape
	assert jnp.array_equal(result, expected_output)


@pytest.mark.parametrize(
	"input_array, expected_rgb",
	[
		# 1 channel -> grayscale repeated
		(jnp.array([0.5]), jnp.array([0.5, 0.5, 0.5])),
		(jnp.array([[0.1], [0.9]]), jnp.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])),
		# 2 channels -> hue, saturation (value=1.0)
		(jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0, 0.0])),  # Red
		(jnp.array([1 / 3, 1.0]), jnp.array([0.0, 1.0, 0.0])),  # Green
		(
			jnp.array([[0.5, 1.0], [0.0, 0.0]]),
			jnp.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
		),  # Cyan, White
		# 3 channels -> RGB direct
		(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2, 0.3])),
		(
			jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
			jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
		),
		# 4 channels -> last 3 are RGB
		(jnp.array([0.1, 0.2, 0.3, 0.4]), jnp.array([0.2, 0.3, 0.4])),
		(
			jnp.array([[0.9, 0.1, 0.2, 0.3], [0.8, 0.4, 0.5, 0.6]]),
			jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
		),
		# 5 channels -> last 3 are RGB
		(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]), jnp.array([0.3, 0.4, 0.5])),
	],
)
def test_render_array_with_channels_to_rgb(input_array: Array, expected_rgb: Array) -> None:
	"""Test the render_array_with_channels_to_rgb function."""
	result = render_array_with_channels_to_rgb(input_array)
	assert result.shape == expected_rgb.shape
	assert jnp.allclose(result, expected_rgb, atol=1e-6)


# === Test render_array_with_channels_to_rgba ===
@pytest.mark.parametrize(
	"input_array, expected_rgba",
	[
		# 1 channel -> repeated RGBA
		(jnp.array([0.5]), jnp.array([0.5, 0.5, 0.5, 0.5])),
		(jnp.array([[0.1], [0.9]]), jnp.array([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9]])),
		# 2 channels -> [Gray, Alpha]
		(jnp.array([0.8, 0.5]), jnp.array([0.8, 0.8, 0.8, 0.5])),
		(
			jnp.array([[0.2, 1.0], [0.7, 0.0]]),
			jnp.array([[0.2, 0.2, 0.2, 1.0], [0.7, 0.7, 0.7, 0.0]]),
		),
		# 3 channels -> [Hue, Sat, Alpha] (Value=1.0)
		(jnp.array([0.0, 1.0, 0.5]), jnp.array([1.0, 0.0, 0.0, 0.5])),  # Red, 50% alpha
		(jnp.array([1 / 3, 1.0, 1.0]), jnp.array([0.0, 1.0, 0.0, 1.0])),  # Green, 100% alpha
		(
			jnp.array([[0.5, 1.0, 0.2], [0.0, 0.0, 0.8]]),
			jnp.array([[0.0, 1.0, 1.0, 0.2], [1.0, 1.0, 1.0, 0.8]]),
		),  # Cyan (20% alpha), White (80% alpha)
		# 4 channels -> RGBA direct
		(jnp.array([0.1, 0.2, 0.3, 0.4]), jnp.array([0.1, 0.2, 0.3, 0.4])),
		(
			jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
			jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
		),
		# 5 channels -> last 4 are RGBA
		(jnp.array([0.9, 0.1, 0.2, 0.3, 0.4]), jnp.array([0.1, 0.2, 0.3, 0.4])),
		(
			jnp.array([[0.9, 0.1, 0.2, 0.3, 0.4], [0.8, 0.5, 0.6, 0.7, 0.8]]),
			jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
		),
	],
)
def test_render_array_with_channels_to_rgba(input_array: Array, expected_rgba: Array) -> None:
	"""Test the render_array_with_channels_to_rgba function."""
	result = render_array_with_channels_to_rgba(input_array)
	assert result.shape == expected_rgba.shape
	assert jnp.allclose(result, expected_rgba, atol=1e-6)
