"""Encoder module.

An encoder is an `nnx.Module` mapping observation frames to named per-frame feature
series. The encoder is the experiment's only weight carrier: evaluation enriches each
phenotype's `series` with its output, and fitness/descriptor are weightless reductions
over named series. AURORA is then literal: the search trains the encoder, and both the
descriptor (mean latent) and an unsupervised fitness (latent variance) are statistics of
it. The base `Encoder` is the no-op — no weights, no series, window 1 — and `refit` on a
fixed encoder returns it unchanged, so every experiment runs the same code path. Each
encoder class sits beside its config; `EncoderConfig` is their discriminated union on
`name`.

Pose invariance, two levers:

- `augment` (trainable encoders): each training batch sample is randomly transformed by
  the named symmetry-group augmentations — `d4` (the dihedral group: flips and quarter
  rotations, exact on a square grid) and `roll` (torus translation, exact). Only the
  dynamics' own symmetries are used; photographic policies (color, crop, scale) would
  corrupt the physical state.
- `invariant` (any encoder): `encode` averages the features over the D4 orbit, making
  the emitted series *exactly* invariant to flips and quarter rotations — a novelty
  search cannot farm them, unlike with augmentation alone. The orbit is unrolled
  sequentially to keep peak memory at one forward pass.
"""

from functools import partial
from typing import Annotated, ClassVar, Literal, override

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from cax.nn.vae import VAE, vae_loss

from .complex_system import ComplexSystem
from .vgg import VGG16

Augmentation = Literal["d4", "roll"]

# The dihedral group D4 acting on one (H, W, C) frame: 4 rotations x optional flip
D4 = tuple(
    partial(
        lambda frame, k, flip: jnp.rot90(frame[::-1] if flip else frame, k),
        k=k,
        flip=flip,
    )
    for flip in (False, True)
    for k in range(4)
)


def augment_batch(key: Array, batch: Array, augment: tuple[Augmentation, ...]) -> Array:
    """Randomly transform each (H, W, C) sample of a batch by the augmentations."""
    key_d4, key_h, key_w = jax.random.split(key, 3)
    num_samples, height, width = batch.shape[:3]
    if "d4" in augment:
        index = jax.random.randint(key_d4, (num_samples,), 0, len(D4))
        batch = jax.vmap(lambda frame, i: jax.lax.switch(i, D4, frame))(batch, index)
    if "roll" in augment:
        shifts_h = jax.random.randint(key_h, (num_samples,), 0, height)
        shifts_w = jax.random.randint(key_w, (num_samples,), 0, width)
        batch = jax.vmap(lambda frame, h, w: jnp.roll(frame, (h, w), axis=(0, 1)))(
            batch, shifts_h, shifts_w
        )
    return batch


class Encoder(nnx.Module):
    """Maps observation frames to named feature series; the base class is the no-op."""

    series: ClassVar[str | None] = None
    window: int = 1
    invariant: bool = False

    def embed(self, frames: Array) -> Array:
        """Embed frames to a feature series with shape `(window, feature_size)`.

        Args:
            frames: RGB frames with dtype uint8 and shape `(window, H, W, 3)`.

        """
        raise NotImplementedError

    def encode(self, frames: Array) -> dict[str, Array]:
        """Encode frames to named feature series (D4-orbit-averaged if `invariant`)."""
        if self.series is None:
            return {}
        if self.invariant:
            # Unrolled orbit: 8 sequential embeds keep peak memory at one forward pass;
            # only the (window, feature_size) outputs are stacked
            features = jnp.mean(
                jnp.stack(
                    [self.embed(jax.vmap(transform)(frames)) for transform in D4]
                ),
                axis=0,
            )
        else:
            features = self.embed(frames)
        return {self.series: features}

    def refit(self, key: Array, observations: Array, *, valid: Array) -> "Encoder":
        """Return the encoder for the next block; fixed encoders return themselves."""
        return self


class CNNEncoder(Encoder):
    """Fixed pretrained encoder: jittable VGG16 features, spatially pooled per frame.

    Emits the `vgg` series. Torchvision's ImageNet VGG16 weights are converted once into
    `nnx.Conv` modules at construction (following the texture NCA example); the forward
    pass afterwards is pure JAX and fully jittable. Requires `torch` and `torchvision`
    at construction only (CPU builds suffice — they are just weight readers).
    """

    series: ClassVar[str] = "vgg"

    def __init__(self, *, layer: int, window: int, invariant: bool, rngs: nnx.Rngs):
        """Initialize the pretrained encoder.

        Args:
            layer: Index into VGG16's feature stack at which activations are taken
                (e.g. 11 is the third block's last relu, 256 channels).
            window: Number of final frames encoded, excluding the developmental
            transient.
            invariant: Whether `encode` averages features over the D4 orbit.
            rngs: rng key (placeholder — weights are overwritten with VGG16's).

        """
        self.vgg = VGG16(layer=layer, rngs=rngs)
        self.window = window
        self.invariant = invariant
        self.feature_size = self.vgg.feature_size

    @override
    def embed(self, frames: Array) -> Array:
        """Embed frames to spatially pooled VGG16 features."""
        return jnp.mean(self.vgg(frames), axis=(1, 2))


class CNNEncoderConfig(BaseModel):
    """Config of `CNNEncoder` (fixed pretrained VGG16 features, the `vgg` series)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series: ClassVar[str] = "vgg"
    trainable: ClassVar[bool] = False

    name: Literal["vgg"] = "vgg"
    layer: PositiveInt = 11
    window: PositiveInt = 32
    invariant: bool = True

    def build(self, key: Array, cs: ComplexSystem) -> Encoder:
        """Build the configured encoder."""
        return CNNEncoder(
            layer=self.layer,
            window=self.window,
            invariant=self.invariant,
            rngs=nnx.Rngs(key),
        )


class VAEEncoder(Encoder):
    """Learned encoder: per-frame VAE latent means (AURORA-style), the `latent` series.

    The feature series is the deterministic encoder mean of each frame. `refit` rebuilds
    the VAE from scratch and trains it on the population's observations with the
    module's own training recipe.
    """

    series: ClassVar[str] = "latent"

    def __init__(
        self,
        *,
        spatial_dims: tuple[int, int],
        features: tuple[int, ...],
        latent_size: int,
        window: int,
        num_steps: int,
        batch_size: int,
        learning_rate: float,
        grad_clip: float,
        augment: tuple[Augmentation, ...],
        invariant: bool,
        padding: str,
        rngs: nnx.Rngs,
    ):
        """Initialize the VAE encoder.

        Args:
            spatial_dims: Spatial dimensions of the frames.
            features: Feature sizes of the VAE encoder, starting with the input
            channels.
            latent_size: Size of the latent space, i.e. the feature size.
            window: Number of final frames encoded, excluding the developmental
            transient.
            num_steps: Gradient steps per `fit`.
            batch_size: Batch size per gradient step.
            learning_rate: Adam learning rate.
            grad_clip: Global-norm gradient clipping (0 disables).
            augment: Symmetry-group augmentations applied to training batches.
            invariant: Whether `encode` averages features over the D4 orbit.
            padding: Convolution padding ("CIRCULAR" matches the toroidal world).
            rngs: rng key.

        """
        self.vae = VAE(
            spatial_dims=spatial_dims,
            features=features,
            latent_size=latent_size,
            padding=padding,
            rngs=rngs,
        )
        self.spatial_dims = spatial_dims
        self.features = features
        self.latent_size = latent_size
        self.window = window
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.augment = augment
        self.invariant = invariant
        self.padding = padding

    @override
    def embed(self, frames: Array) -> Array:
        """Embed frames to per-frame latent means."""
        mean, _ = self.vae.encoder(frames.astype(jnp.float32) / 255.0)
        return mean

    @override
    def refit(self, key: Array, observations: Array, *, valid: Array) -> "VAEEncoder":
        """Rebuild the VAE from scratch and train it on the observations (AURORA).

        Each training step samples a batch of (individual, frame) pairs uniformly among
        valid individuals, following the official Leniabreeder training scheme, then
        applies the configured augmentations.

        Args:
            key: rng key.
            observations: RGB frames with dtype uint8 and shape `(N, window, H, W, 3)`.
            valid: Boolean array with shape `(N,)`; invalid individuals are never
            sampled.

        """
        key_init, key_fit = jax.random.split(key)
        encoder_fn = VAEEncoder(
            spatial_dims=self.spatial_dims,
            features=self.features,
            latent_size=self.latent_size,
            window=self.window,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            grad_clip=self.grad_clip,
            augment=self.augment,
            invariant=self.invariant,
            padding=self.padding,
            rngs=nnx.Rngs(key_init),
        )
        tx = optax.adam(self.learning_rate)
        if self.grad_clip:
            tx = optax.chain(optax.clip_by_global_norm(self.grad_clip), tx)
        optimizer = nnx.Optimizer(encoder_fn.vae, tx, wrt=nnx.Param)
        keys = jax.random.split(key_fit, self.num_steps)
        fit_scan(
            encoder_fn.vae,
            optimizer,
            observations,
            valid,
            keys,
            batch_size=self.batch_size,
            augment=self.augment,
        )
        return encoder_fn


class VAEEncoderConfig(BaseModel):
    """Config of `VAEEncoder`: learned latent features (AURORA), the `latent` series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series: ClassVar[str] = "latent"
    trainable: ClassVar[bool] = True

    name: Literal["vae"] = "vae"
    latent_size: PositiveInt = 8
    features: tuple[int, ...] = (3, 16, 32, 32)
    learning_rate: PositiveFloat = 1e-3
    grad_clip: NonNegativeFloat = 1.0
    num_steps: PositiveInt = 8192
    batch_size: PositiveInt = 256
    window: PositiveInt = 32
    augment: tuple[Augmentation, ...] = ("d4", "roll")
    invariant: bool = True
    padding: Literal["SAME", "CIRCULAR"] = "SAME"

    def build(self, key: Array, cs: ComplexSystem) -> VAEEncoder:
        """Build the configured encoder."""
        return VAEEncoder(
            spatial_dims=cs.spatial_dims,
            features=self.features,
            latent_size=self.latent_size,
            window=self.window,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            grad_clip=self.grad_clip,
            augment=self.augment,
            invariant=self.invariant,
            padding=self.padding,
            rngs=nnx.Rngs(key),
        )


class NoEncoderConfig(BaseModel):
    """Config of the no-op encoder (`encoder: null` in YAML): no weights, no series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series: ClassVar[None] = None
    trainable: ClassVar[bool] = False
    window: ClassVar[int] = 1

    name: Literal["none"] = "none"

    def build(self, key: Array, cs: ComplexSystem) -> Encoder:
        """Build the no-op encoder."""
        return Encoder()


EncoderConfig = Annotated[
    VAEEncoderConfig | CNNEncoderConfig | NoEncoderConfig, Field(discriminator="name")
]


@partial(nnx.jit, static_argnames=("batch_size", "augment"))
def fit_scan(
    vae: VAE,
    optimizer: nnx.Optimizer,
    observations: Array,
    valid: Array,
    keys: Array,
    *,
    batch_size: int,
    augment: tuple[Augmentation, ...],
) -> Array:
    """Run all VAE gradient steps in one compiled scan."""
    num_individuals, num_frames = observations.shape[:2]
    # Sample valid individuals; before any exist, fall back to uniform
    p = jnp.where(
        jnp.any(valid), valid / jnp.maximum(jnp.sum(valid), 1), 1.0 / num_individuals
    )

    @partial(nnx.scan, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(
        carry: tuple[VAE, nnx.Optimizer], key: Array
    ) -> tuple[tuple[VAE, nnx.Optimizer], Array]:
        vae, optimizer = carry
        key_individual, key_frame, key_augment = jax.random.split(key, 3)
        individual_idx = jax.random.choice(
            key_individual, num_individuals, (batch_size,), p=p
        )
        frame_idx = jax.random.randint(key_frame, (batch_size,), 0, num_frames)
        batch = observations[individual_idx, frame_idx]
        batch = augment_batch(key_augment, batch, augment).astype(jnp.float32) / 255.0

        def loss_fn(vae: VAE) -> Array:
            logits, mean, logvar = vae(batch)
            return vae_loss(logits, batch, mean, logvar)

        loss, grads = nnx.value_and_grad(loss_fn)(vae)
        optimizer.update(vae, grads)
        return (vae, optimizer), loss

    _, losses = scan_step((vae, optimizer), keys)
    return losses
