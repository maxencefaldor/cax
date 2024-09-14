"""Variational Autoencoder (VAE) implementation using JAX and Flax."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class Encoder(nnx.Module):
	"""Encoder module for the VAE."""

	features: Sequence[int]
	latent_size: int
	convs: Sequence[nnx.Conv]
	linear: nnx.Linear
	mean: nnx.Linear
	logvar: nnx.Linear

	def __init__(self, spatial_dims: Sequence[int], features: Sequence[int], latent_size: int, rngs: nnx.Rngs):
		"""Initialize the Encoder module.

		Args:
			spatial_dims: Spatial dimensions of the input.
			features: Sequence of feature sizes for convolutional layers.
			latent_size: Size of the latent space.
			rngs: Random number generators.

		"""
		super().__init__()
		self.features = features
		self.latent_size = latent_size

		self.convs = []
		for in_features, out_features in zip(self.features[:-1], self.features[1:]):
			self.convs.append(
				nnx.Conv(
					in_features=in_features,
					out_features=out_features,
					kernel_size=(3, 3),
					strides=(2, 2),
					padding="SAME",
					rngs=rngs,
				)
			)

		flattened_size = spatial_dims[0] * spatial_dims[1] * self.features[-1]
		for _ in range(len(self.features) - 1):
			flattened_size //= 4

		self.linear = nnx.Linear(in_features=flattened_size, out_features=flattened_size, rngs=rngs)
		self.mean = nnx.Linear(in_features=flattened_size, out_features=self.latent_size, rngs=rngs)
		self.logvar = nnx.Linear(in_features=flattened_size, out_features=self.latent_size, rngs=rngs)

	def __call__(self, x):
		"""Forward pass of the encoder.

		Args:
			x: Input tensor.

		Returns:
			Tuple of mean and log variance of the latent distribution.

		"""
		for conv in self.convs:
			x = jax.nn.relu(conv(x))
		x = x.reshape(*x.shape[:-3], -1)
		x = jax.nn.relu(self.linear(x))
		mean = self.mean(x)
		logvar = self.logvar(x)
		return mean, logvar


class Decoder(nnx.Module):
	"""Decoder module for the VAE."""

	features: Sequence[int]
	latent_size: int
	convs: Sequence[nnx.ConvTranspose]
	linear: nnx.Linear
	_spatial_dims: tuple[int, int]

	def __init__(self, spatial_dims: Sequence[int], features: Sequence[int], latent_size: int, rngs: nnx.Rngs):
		"""Initialize the Decoder module.

		Args:
			spatial_dims: Spatial dimensions of the output.
			features: Sequence of feature sizes for transposed convolutional layers.
			latent_size: Size of the latent space.
			rngs: Random number generators.

		"""
		super().__init__()
		self.features = features
		self.latent_size = latent_size

		self._spatial_dims = tuple(dim // (2 ** (len(self.features) - 1)) for dim in spatial_dims[:2])

		flattened_size = self._spatial_dims[0] * self._spatial_dims[1] * self.features[0]

		self.linear = nnx.Linear(in_features=self.latent_size, out_features=flattened_size, rngs=rngs)

		self.convs = []
		for in_features, out_features in zip(self.features[:-1], self.features[1:]):
			self.convs.append(
				nnx.ConvTranspose(
					in_features=in_features,
					out_features=out_features,
					kernel_size=(3, 3),
					strides=(2, 2),
					padding="SAME",
					rngs=rngs,
				)
			)

	def __call__(self, z):
		"""Forward pass of the decoder.

		Args:
			z: Latent vector.

		Returns:
			Reconstructed output tensor.

		"""
		x = jax.nn.relu(self.linear(z))
		x = x.reshape(*x.shape[:-1], *self._spatial_dims, self.features[0])
		for conv in self.convs[:-1]:
			x = jax.nn.relu(conv(x))
		x = self.convs[-1](x)
		return x


class VAE(nnx.Module):
	"""Variational Autoencoder (VAE) module."""

	encoder: Encoder
	decoder: Decoder

	def __init__(self, spatial_dims: tuple[int, int], features: Sequence[int], latent_size: int, rngs: nnx.Rngs):
		"""Initialize the VAE module.

		Args:
			spatial_dims: Spatial dimensions of the input/output.
			features: Sequence of feature sizes for encoder and decoder.
			latent_size: Size of the latent space.
			rngs: Random number generators.

		"""
		super().__init__()
		self.encoder = Encoder(spatial_dims=spatial_dims, features=features, latent_size=latent_size, rngs=rngs)
		self.decoder = Decoder(spatial_dims=spatial_dims, features=features[::-1], latent_size=latent_size, rngs=rngs)

	def reparameterize(self, mean, logvar, key):
		"""Perform the reparameterization trick.

		Args:
			mean: Mean of the latent distribution.
			logvar: Log variance of the latent distribution.
			key: Random key for sampling.

		Returns:
			Sampled latent vector.

		"""
		eps = jax.random.normal(key, shape=mean.shape)
		return eps * jnp.exp(logvar * 0.5) + mean

	def encode(self, x, key):
		"""Encode input to latent space.

		Args:
			x: Input tensor.
			key: Random key for sampling.

		Returns:
			Tuple of sampled latent vector, mean, and log variance.

		"""
		mean, logvar = self.encoder(x)
		return self.reparameterize(mean, logvar, key), mean, logvar

	def decode(self, z):
		"""Decode latent vector to output space.

		Args:
			z: Latent vector.

		Returns:
			Reconstructed output tensor.

		"""
		return self.decoder(z)

	def generate(self, z):
		"""Generate output from latent vector.

		Args:
			z: Latent vector.

		Returns:
			Generated output tensor.

		"""
		return jax.nn.sigmoid(self.decoder(z))

	def __call__(self, x, key):
		"""Forward pass of the VAE.

		Args:
			x: Input tensor.
			key: Random key for sampling.

		Returns:
			Tuple of reconstructed logits, mean, and log variance.

		"""
		z, mean, logvar = self.encode(x, key)
		logits = self.decode(z)
		return logits, mean, logvar


@jax.jit
def kl_divergence(mean, logvar):
	"""Compute KL divergence between latent distribution and standard normal.

	Args:
		mean: Mean of the latent distribution.
		logvar: Log variance of the latent distribution.

	Returns:
		KL divergence value.

	"""
	return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
	"""Compute binary cross-entropy loss with logits.

	Args:
		logits: Predicted logits.
		labels: True labels.

	Returns:
		Binary cross-entropy loss.

	"""
	logits = jax.nn.log_sigmoid(logits)
	return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def vae_loss(logits, targets, mean, logvar):
	"""Compute VAE loss.

	Args:
		logits: Predicted logits.
		targets: True targets.
		mean: Mean of the latent distribution.
		logvar: Log variance of the latent distribution.

	Returns:
		Total VAE loss.

	"""
	bce_loss = binary_cross_entropy_with_logits(logits, targets).mean()
	kld_loss = kl_divergence(mean, logvar).mean()
	return bce_loss + kld_loss
