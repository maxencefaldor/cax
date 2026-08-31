"""VGG16 feature module.

A jittable VGG16 feature stack: torchvision's ImageNet weights are converted once into
`nnx.Conv` modules at construction (pattern from the texture NCA example); the forward
pass afterwards is pure JAX. torch/torchvision (CPU builds suffice) are needed only at
construction, as weight readers. Used by the pretrained descriptor (pooled features) and
the deepdream-style fitness (feature activation maximization).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


class VGG16(nnx.Module):
	"""Jittable VGG16 feature stack up to a chosen layer."""

	def __init__(self, *, layer: int, rngs: nnx.Rngs):
		"""Initialize from torchvision's pretrained VGG16.

		Args:
			layer: Index into VGG16's feature stack at which activations are taken
				(e.g. 11 is the third block's last relu, 256 channels).
			rngs: rng key (placeholder — weights are overwritten with VGG16's).

		"""
		import torch
		from torchvision import models

		torch_features = models.vgg16(weights="IMAGENET1K_V1").features.eval()

		convs = []
		layer_types = []
		for i in range(layer + 1):
			module = torch_features[i]
			if isinstance(module, torch.nn.Conv2d):
				conv = nnx.Conv(
					in_features=module.in_channels,
					out_features=module.out_channels,
					kernel_size=(module.kernel_size[0], module.kernel_size[1]),
					strides=(module.stride[0], module.stride[1]),
					padding=(
						(module.padding[0], module.padding[0]),
						(module.padding[1], module.padding[1]),
					),
					use_bias=module.bias is not None,
					rngs=rngs,
				)
				kernel = module.weight.detach().cpu().numpy()
				conv.kernel.value = jnp.asarray(kernel.transpose(2, 3, 1, 0))
				if module.bias is not None:
					conv.bias.value = jnp.asarray(module.bias.detach().cpu().numpy())
				convs.append(conv)
				layer_types.append("conv")
			elif isinstance(module, torch.nn.ReLU):
				layer_types.append("relu")
			elif isinstance(module, torch.nn.MaxPool2d):
				layer_types.append("maxpool")
			else:
				raise ValueError(f"Unexpected VGG16 layer type: {type(module)}")

		self.convs = nnx.List(convs)
		self.layer_types = layer_types
		self.feature_size = convs[-1].out_features

	def __call__(self, frames: Array) -> Array:
		"""Compute feature activations for a batch of frames.

		Args:
			frames: RGB frames with dtype uint8 and shape `(..., H, W, 3)`.

		Returns:
			Activations with shape `(..., h, w, feature_size)`.

		"""
		mean = jnp.array([0.485, 0.456, 0.406])
		std = jnp.array([0.229, 0.224, 0.225])
		x = (frames.astype(jnp.float32) / 255.0 - mean) / std

		conv_idx = 0
		for layer_type in self.layer_types:
			if layer_type == "conv":
				x = self.convs[conv_idx](x)
				conv_idx += 1
			elif layer_type == "relu":
				x = jax.nn.relu(x)
			elif layer_type == "maxpool":
				x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

		return x
