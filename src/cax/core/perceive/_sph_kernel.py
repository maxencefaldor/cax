"""The SPH neighborhood sums, fused into Pallas kernels.

The four quantities a particle perceives --- density, the neighborhood-average state,
the state gradient and the density gradient --- are four sums over the *same* pairs.
Written as array operations each one walks an array with an entry per pair, so a few
thousand particles move gigabytes through memory to produce a few megabytes of answer.
Here they accumulate together, one tile of neighbors at a time, in registers, and only
the answers are ever written. It is the tiling attention kernels use, for the same
reason.

The saving is in *space*, not in time complexity: every pair is still visited, but
nothing with an entry per pair is ever built. That is what lets a cloud grow past the
point where the array version simply runs out of memory.

Two things follow from writing a kernel. It runs on GPU only, so `SPHPerceive` keeps its
array implementation and reaches for this one on request. And a kernel is opaque to
automatic differentiation, so the backward pass is derived by hand below --- the part
where a mistake would train quietly rather than fail.

Coordinates travel as separate components rather than as vectors: a tiled kernel wants
plain two-dimensional tiles, and keeping the spatial axis out of the arrays is what lets
every accumulator stay one.

Nothing here is specialized to a particular cloud size. The tile shape is chosen from
the particle count, and a count the tiles do not divide is padded and masked, the way an
attention kernel handles a sequence length that is not a multiple of its block.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

# How many blocks it takes to keep a large GPU busy. Below this the multiprocessors sit
# idle and the kernel loses to plain array operations; above it, a bigger query tile is
# worth more than more blocks, because each tile of neighbors is then loaded for more
# queries. Both effects are real and they pull in opposite directions, which is why the
# tile is chosen from the cloud size rather than fixed.
TARGET_BLOCKS = 256
BLOCK_NEIGHBORS = 64

# How deep the compiler pipelines the loop over neighbor tiles: with more stages it
# starts fetching the next tile while the current one is still being reduced, which is
# what a loop of dependent global loads wants.
NUM_STAGES = 3

# The backward is a different kernel with a different appetite. It carries four
# accumulators instead of one and reads ten arrays instead of four, so it runs out of
# registers at a tile the forward is happy with, and it wants a smaller one. Attention
# kernels split these the same way, tuning the forward and the two backward passes
# separately.
BACKWARD_BLOCK_QUERIES = 32


def _num_warps(channel_size: int) -> int:
    """Choose the warp count from the channel width, which sets the tile's area."""
    return 4 if channel_size <= 64 else 8


def _tile_shape(num_particles: int) -> int:
    """Choose how many queries a block answers for, from the size of the cloud.

    Args:
        num_particles: Number of particles in the cloud.

    Returns:
        The number of queries per block.

    """
    block_queries = 16
    while block_queries < 128 and num_particles > block_queries * TARGET_BLOCKS:
        block_queries *= 2
    return block_queries


def _live(k: Array, block_neighbors: int, num_particles: int) -> Array:
    """Mark which neighbors in a tile are real, so padding contributes nothing."""
    index = k * block_neighbors + jnp.arange(block_neighbors)
    return (index < num_particles)[None, :]


def _pad_to(array: Array, length: int) -> Array:
    """Extend the leading axis to `length`, leaving the new entries at zero."""
    if array.shape[0] == length:
        return array
    padding = [(0, length - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, padding)


def _wrap(displacement: Array, period: float | None) -> Array:
    """Take the shortest displacement on a torus."""
    if period is None:
        return displacement
    # floor(x + 1/2) stands in for round, which Triton does not lower.
    return displacement - period * jnp.floor(displacement / period + 0.5)


def _weights(dx: Array, dy: Array, support_radius: float) -> tuple[Array, Array]:
    """Poly6 weight and the scalar factor of the Spiky gradient, for a tile of pairs."""
    square_distance = dx * dx + dy * dy
    inside = support_radius**2 - square_distance
    weight = jnp.where(
        inside > 0.0, (4.0 / (math.pi * support_radius**8)) * inside**3, 0.0
    )

    distance = jnp.sqrt(square_distance + 1e-18)
    margin = support_radius - distance
    scale = jnp.where(
        (margin > 0.0) & (distance > 1e-9),
        -(30.0 / (math.pi * support_radius**5)) * margin * margin / distance,
        0.0,
    )
    return weight, scale


def _queries(block_queries: int) -> object:
    """Return the slice of particles this block answers for."""
    return pl.ds(pl.program_id(0) * block_queries, block_queries)


def _density_kernel(
    x_ref,
    y_ref,
    density_ref,
    *,
    support_radius,
    mass,
    period,
    num_tiles,
    block_queries,
    num_particles,
):
    """Sum each particle's kernel weights over every tile of neighbors."""
    queries = _queries(block_queries)
    own_x, own_y = x_ref[queries][:, None], y_ref[queries][:, None]

    def tile(k, total):
        span = pl.ds(k * BLOCK_NEIGHBORS, BLOCK_NEIGHBORS)
        dx = _wrap(own_x - x_ref[span][None, :], period)
        dy = _wrap(own_y - y_ref[span][None, :], period)
        weight, _ = _weights(dx, dy, support_radius)
        weight = jnp.where(_live(k, BLOCK_NEIGHBORS, num_particles), weight, 0.0)
        return total + jnp.sum(weight, axis=-1)

    density_ref[...] = mass * jax.lax.fori_loop(
        0, num_tiles, tile, jnp.zeros(block_queries, density_ref.dtype)
    )


def _moments_kernel(
    x_ref,
    y_ref,
    feature_ref,
    volume_ref,
    average_ref,
    gradient_x_ref,
    gradient_y_ref,
    density_gradient_x_ref,
    density_gradient_y_ref,
    *,
    support_radius,
    mass,
    period,
    num_tiles,
    channel_size,
    block_queries,
    num_particles,
):
    """Accumulate the average, state gradient and density gradient in one pass."""
    queries = _queries(block_queries)
    own_x, own_y = x_ref[queries][:, None], y_ref[queries][:, None]
    own_feature = feature_ref[queries]

    def tile(k, carry):
        average, first_x, first_y, weighted_x, weighted_y, raw_x, raw_y = carry
        span = pl.ds(k * BLOCK_NEIGHBORS, BLOCK_NEIGHBORS)
        dx = _wrap(own_x - x_ref[span][None, :], period)
        dy = _wrap(own_y - y_ref[span][None, :], period)
        feature = feature_ref[span]
        volume = volume_ref[span][None, :]

        weight, scale = _weights(dx, dy, support_radius)
        live = _live(k, BLOCK_NEIGHBORS, num_particles)
        weight = jnp.where(live, weight, 0.0)
        scale = jnp.where(live, scale, 0.0)
        component_x, component_y = scale * dx, scale * dy

        average += (weight * volume) @ feature

        # The gradient's difference form, split so that no array with one entry per pair
        # *and* per channel is ever built:
        #   sum_j v_j (f_j - f_i) (x) grad W_ij
        # = sum_j v_j f_j (x) grad W_ij  -  f_i (x) sum_j v_j grad W_ij
        first_x += (component_x * volume) @ feature
        first_y += (component_y * volume) @ feature
        weighted_x += jnp.sum(component_x * volume, axis=-1)
        weighted_y += jnp.sum(component_y * volume, axis=-1)
        raw_x += jnp.sum(component_x, axis=-1)
        raw_y += jnp.sum(component_y, axis=-1)
        return average, first_x, first_y, weighted_x, weighted_y, raw_x, raw_y

    channels = jnp.zeros((block_queries, channel_size), own_feature.dtype)
    scalars = jnp.zeros(block_queries, own_feature.dtype)
    average, first_x, first_y, weighted_x, weighted_y, raw_x, raw_y = jax.lax.fori_loop(
        0,
        num_tiles,
        tile,
        (channels, channels, channels, scalars, scalars, scalars, scalars),
    )

    average_ref[...] = average
    gradient_x_ref[...] = first_x - own_feature * weighted_x[:, None]
    gradient_y_ref[...] = first_y - own_feature * weighted_y[:, None]
    density_gradient_x_ref[...] = mass * raw_x
    density_gradient_y_ref[...] = mass * raw_y


@functools.partial(jax.jit, static_argnames=("support_radius", "mass", "period"))
def _forward(
    position: Array,
    feature: Array,
    *,
    support_radius: float,
    mass: float,
    period: float | None,
) -> tuple[Array, Array, Array, Array]:
    """Perceive every particle's neighborhood in one pass over the pairs."""
    num_particles = position.shape[0]
    channel_size = feature.shape[-1]

    # The tiling is static, so a cloud the tiles do not divide is padded out and the
    # extra entries are masked away inside the kernel. Nothing is specialized to one
    # size.
    block_queries = _tile_shape(num_particles)
    padded = math.lcm(block_queries, BLOCK_NEIGHBORS)
    padded = -(-num_particles // padded) * padded
    position, feature = _pad_to(position, padded), _pad_to(feature, padded)

    x, y = position[:, 0], position[:, 1]
    grid = (padded // block_queries,)
    num_tiles = padded // BLOCK_NEIGHBORS
    triton = plt.CompilerParams(
        num_warps=_num_warps(channel_size), num_stages=NUM_STAGES
    )
    shape = functools.partial(
        dict,
        support_radius=support_radius,
        mass=mass,
        period=period,
        num_tiles=num_tiles,
        block_queries=block_queries,
        num_particles=num_particles,
    )

    query_spec = pl.BlockSpec((block_queries,), lambda i: (i,))
    whole_spec = pl.BlockSpec(memory_space=pl.ANY)

    density = pl.pallas_call(
        functools.partial(_density_kernel, **shape()),
        grid=grid,
        compiler_params=triton,
        in_specs=[whole_spec, whole_spec],
        out_specs=query_spec,
        out_shape=jax.ShapeDtypeStruct((padded,), position.dtype),
    )(x, y)

    # Each neighbor is weighted by the volume *it* stands for, so the density has to be
    # complete before the rest can start. Hence two passes, not one. Padded particles
    # are given no volume at all, so they cannot weigh on anything even before the mask.
    volume = mass / jnp.maximum(density, 1e-8)
    volume = jnp.where(jnp.arange(padded) < num_particles, volume, 0.0)

    channel_spec = pl.BlockSpec((block_queries, channel_size), lambda i: (i, 0))
    average, gradient_x, gradient_y, density_gradient_x, density_gradient_y = (
        pl.pallas_call(
            functools.partial(_moments_kernel, channel_size=channel_size, **shape()),
            grid=grid,
            compiler_params=triton,
            in_specs=[whole_spec] * 4,
            out_specs=[
                channel_spec,
                channel_spec,
                channel_spec,
                query_spec,
                query_spec,
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded, channel_size), position.dtype),
                jax.ShapeDtypeStruct((padded, channel_size), position.dtype),
                jax.ShapeDtypeStruct((padded, channel_size), position.dtype),
                jax.ShapeDtypeStruct((padded,), position.dtype),
                jax.ShapeDtypeStruct((padded,), position.dtype),
            ],
        )(x, y, feature, volume)
    )

    gradient = jnp.stack([gradient_x, gradient_y], axis=-1)
    density_gradient = jnp.stack([density_gradient_x, density_gradient_y], axis=-1)
    return (
        density[:num_particles],
        average[:num_particles],
        gradient[:num_particles],
        density_gradient[:num_particles],
    )


def _terms(
    rx: Array, ry: Array, radius: float
) -> tuple[Array, Array, Array, Array, Array]:
    """Evaluate the pair kernels and their derivatives, for a tile of pairs."""
    square_distance = rx * rx + ry * ry
    inside = radius**2 - square_distance
    live = inside > 0.0
    weight = jnp.where(live, (4.0 / (math.pi * radius**8)) * inside**3, 0.0)
    poly_scale = jnp.where(live, -(24.0 / (math.pi * radius**8)) * inside**2, 0.0)

    distance = jnp.sqrt(square_distance + 1e-18)
    margin = radius - distance
    spiky_live = (margin > 0.0) & (distance > 1e-9)
    spiky_scale = jnp.where(
        spiky_live, -(30.0 / (math.pi * radius**5)) * margin * margin / distance, 0.0
    )
    spiky_derivative = jnp.where(
        spiky_live,
        (30.0 / (math.pi * radius**5))
        * margin
        * (margin + 2.0 * distance)
        / (distance * distance),
        0.0,
    )
    return weight, poly_scale, spiky_scale, spiky_derivative, distance


def _vjp_kernel(
    x_ref,
    y_ref,
    feature_ref,
    volume_ref,
    cotangent_density_ref,
    cotangent_average_ref,
    cotangent_gradient_x_ref,
    cotangent_gradient_y_ref,
    cotangent_density_gradient_x_ref,
    cotangent_density_gradient_y_ref,
    d_x_ref,
    d_y_ref,
    d_feature_ref,
    d_volume_ref,
    *,
    support_radius,
    mass,
    period,
    num_tiles,
    block_queries,
    num_particles,
    wants_position,
    wants_feature,
):
    """Every cotangent as a sum over a particle's own neighbors.

    Where the chain rule asks for the derivative with respect to a *neighbor*,
    antisymmetry of the pair kernels turns what would be a scatter into a second gather
    --- so this has the same shape as the forward and needs no atomics. Each array is
    read twice, once for the particles this block answers for and once per tile of their
    neighbors; the suffixes say which.
    """
    queries = _queries(block_queries)
    own_x, own_y = x_ref[queries][:, None], y_ref[queries][:, None]
    own_feature = feature_ref[queries]
    own_volume = volume_ref[queries][:, None]
    own_density = cotangent_density_ref[queries][:, None]
    own_average = cotangent_average_ref[queries]
    own_gradient_x = cotangent_gradient_x_ref[queries]
    own_gradient_y = cotangent_gradient_y_ref[queries]
    own_density_gradient_x = cotangent_density_gradient_x_ref[queries][:, None]
    own_density_gradient_y = cotangent_density_gradient_y_ref[queries][:, None]

    def tile(k, carry):
        d_x, d_y, d_feature, d_volume = carry
        span = pl.ds(k * BLOCK_NEIGHBORS, BLOCK_NEIGHBORS)
        rx = _wrap(own_x - x_ref[span][None, :], period)
        ry = _wrap(own_y - y_ref[span][None, :], period)
        feature = feature_ref[span]
        volume = volume_ref[span][None, :]
        density = cotangent_density_ref[span][None, :]
        average = cotangent_average_ref[span]
        gradient_x, gradient_y = (
            cotangent_gradient_x_ref[span],
            cotangent_gradient_y_ref[span],
        )
        density_gradient_x = cotangent_density_gradient_x_ref[span][None, :]
        density_gradient_y = cotangent_density_gradient_y_ref[span][None, :]

        weight, poly_scale, spiky_scale, spiky_derivative, distance = _terms(
            rx, ry, support_radius
        )
        # Padded neighbors carry no weight through any of the four routes below.
        live = _live(k, BLOCK_NEIGHBORS, num_particles)
        weight = jnp.where(live, weight, 0.0)
        poly_scale = jnp.where(live, poly_scale, 0.0)
        spiky_scale = jnp.where(live, spiky_scale, 0.0)
        spiky_derivative = jnp.where(live, spiky_derivative, 0.0)
        spiky_x, spiky_y = spiky_scale * rx, spiky_scale * ry

        if wants_feature:
            # --- the feature cotangent, all three terms as gathers -------------------
            # from this particle's neighbors' averages, and from its own
            d_feature += own_volume * (weight @ average)
            # from the gradient: +v_i sum_j (cg_j . G_ji), and -sum_j v_j (cg_i . G_ij)
            d_feature += own_volume * (
                (-spiky_x) @ gradient_x + (-spiky_y) @ gradient_y
            )
            d_feature -= own_gradient_x * jnp.sum(spiky_x * volume, axis=-1)[:, None]
            d_feature -= own_gradient_y * jnp.sum(spiky_y * volume, axis=-1)[:, None]

            # --- what this particle's own volume is worth ----------------------------
            # It scales every neighbor's reading of it, so the cotangent gathers from
            # the neighbors that looked.
            average_onto_own = (average @ own_feature.T).T
            gradient_x_onto_own = (gradient_x @ own_feature.T).T
            gradient_y_onto_own = (gradient_y @ own_feature.T).T
            gradient_x_dot_feature = jnp.sum(gradient_x * feature, axis=-1)[None, :]
            gradient_y_dot_feature = jnp.sum(gradient_y * feature, axis=-1)[None, :]
            d_volume += jnp.sum(weight * average_onto_own, axis=-1)
            d_volume += jnp.sum(
                (-spiky_x) * gradient_x_onto_own + (-spiky_y) * gradient_y_onto_own,
                axis=-1,
            )
            d_volume -= jnp.sum(
                (-spiky_x) * gradient_x_dot_feature
                + (-spiky_y) * gradient_y_dot_feature,
                axis=-1,
            )

        if wants_position:
            # --- the position cotangent ----------------------------------------------
            # the poly6 route: through the average and through the density
            average_onto_own = (average @ own_feature.T).T
            gradient_x_onto_own = (gradient_x @ own_feature.T).T
            gradient_y_onto_own = (gradient_y @ own_feature.T).T
            gradient_x_dot_feature = jnp.sum(gradient_x * feature, axis=-1)[None, :]
            gradient_y_dot_feature = jnp.sum(gradient_y * feature, axis=-1)[None, :]
            forward_pair = (own_average @ feature.T) * volume + mass * own_density
            backward_pair = average_onto_own * own_volume + mass * density
            scalar = (forward_pair + backward_pair) * poly_scale

            # the spiky route: the vector kernel's jacobian, s I + s' r r^T / d
            own_gradient_x_dot_own = jnp.sum(own_gradient_x * own_feature, axis=-1)[
                :, None
            ]
            own_gradient_y_dot_own = jnp.sum(own_gradient_y * own_feature, axis=-1)[
                :, None
            ]
            forward_x = ((own_gradient_x @ feature.T) - own_gradient_x_dot_own) * volume
            forward_y = ((own_gradient_y @ feature.T) - own_gradient_y_dot_own) * volume
            backward_x = (gradient_x_onto_own - gradient_x_dot_feature) * own_volume
            backward_y = (gradient_y_onto_own - gradient_y_dot_feature) * own_volume
            # The neighbor's half of the vector kernel enters with the opposite sign:
            # the displacement flips too, and the two flips cancel for the poly6 route
            # above but not for this one.
            vector_x = (
                forward_x
                - backward_x
                + mass * (own_density_gradient_x - density_gradient_x)
            )
            vector_y = (
                forward_y
                - backward_y
                + mass * (own_density_gradient_y - density_gradient_y)
            )

            radial = (vector_x * rx + vector_y * ry) * spiky_derivative / distance
            d_x += jnp.sum(scalar * rx + vector_x * spiky_scale + rx * radial, axis=-1)
            d_y += jnp.sum(scalar * ry + vector_y * spiky_scale + ry * radial, axis=-1)

        return d_x, d_y, d_feature, d_volume

    scalars = jnp.zeros(block_queries, d_x_ref.dtype)
    d_x, d_y, d_feature, d_volume = jax.lax.fori_loop(
        0, num_tiles, tile, (scalars, scalars, jnp.zeros_like(own_feature), scalars)
    )
    d_x_ref[...], d_y_ref[...] = d_x, d_y
    d_feature_ref[...], d_volume_ref[...] = d_feature, d_volume


@functools.partial(
    jax.jit,
    static_argnames=(
        "support_radius",
        "mass",
        "period",
        "wants_position",
        "wants_feature",
    ),
)
def _backward(
    position: Array,
    feature: Array,
    volume: Array,
    cotangents: tuple[Array, Array, Array, Array],
    *,
    support_radius: float,
    mass: float,
    period: float | None,
    wants_position: bool = True,
    wants_feature: bool = True,
) -> tuple[Array, Array, Array]:
    """Position, feature and volume cotangents, in one pass over the pairs."""
    (
        cotangent_density,
        cotangent_average,
        cotangent_gradient,
        cotangent_density_gradient,
    ) = cotangents
    num_particles = position.shape[0]
    channel_size = feature.shape[-1]

    block_queries = min(_tile_shape(num_particles), BACKWARD_BLOCK_QUERIES)
    padded = math.lcm(block_queries, BLOCK_NEIGHBORS)
    padded = -(-num_particles // padded) * padded

    position, feature, volume = (
        _pad_to(position, padded),
        _pad_to(feature, padded),
        _pad_to(volume, padded),
    )
    cotangent_density = _pad_to(cotangent_density, padded)
    cotangent_average = _pad_to(cotangent_average, padded)
    cotangent_gradient = _pad_to(cotangent_gradient, padded)
    cotangent_density_gradient = _pad_to(cotangent_density_gradient, padded)

    x, y = position[:, 0], position[:, 1]
    cgx, cgy = cotangent_gradient[..., 0], cotangent_gradient[..., 1]
    cdgx, cdgy = cotangent_density_gradient[..., 0], cotangent_density_gradient[..., 1]

    query = pl.BlockSpec((block_queries,), lambda i: (i,))
    query_channels = pl.BlockSpec((block_queries, channel_size), lambda i: (i, 0))
    whole = pl.BlockSpec(memory_space=pl.ANY)

    dx, dy, dfeature, dvolume = pl.pallas_call(
        functools.partial(
            _vjp_kernel,
            support_radius=support_radius,
            mass=mass,
            period=period,
            num_tiles=padded // BLOCK_NEIGHBORS,
            block_queries=block_queries,
            num_particles=num_particles,
            wants_position=wants_position,
            wants_feature=wants_feature,
        ),
        grid=(padded // block_queries,),
        compiler_params=plt.CompilerParams(
            num_warps=_num_warps(channel_size), num_stages=NUM_STAGES
        ),
        in_specs=[whole] * 10,
        out_specs=[query, query, query_channels, query],
        out_shape=[
            jax.ShapeDtypeStruct((padded,), position.dtype),
            jax.ShapeDtypeStruct((padded,), position.dtype),
            jax.ShapeDtypeStruct((padded, channel_size), position.dtype),
            jax.ShapeDtypeStruct((padded,), position.dtype),
        ],
    )(
        x,
        y,
        feature,
        volume,
        cotangent_density,
        cotangent_average,
        cgx,
        cgy,
        cdgx,
        cdgy,
    )
    position_cotangent = jnp.stack([dx, dy], axis=-1)[:num_particles]
    return position_cotangent, dfeature[:num_particles], dvolume[:num_particles]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def sph_moments(
    position: Array,
    feature: Array,
    support_radius: float,
    mass: float,
    period: float | None,
) -> tuple[Array, Array, Array, Array]:
    """Density, neighborhood average, state gradient and density gradient.

    Args:
        position: Positions with shape `(num_particles, 2)`.
        feature: States with shape `(num_particles, channel_size)`.
        support_radius: Distance beyond which particles do not interact.
        mass: What each particle weighs.
        period: Size of the periodic domain, or None for an unbounded one.

    Returns:
        The density, the neighborhood average, the state gradient with shape
            `(num_particles, channel_size, 2)`, and the density gradient.

    """
    return _forward(
        position, feature, support_radius=support_radius, mass=mass, period=period
    )


def _moments_fwd(position, feature, support_radius, mass, period):
    outputs = _forward(
        position, feature, support_radius=support_radius, mass=mass, period=period
    )
    return outputs, (position, feature, outputs[0])


def _moments_bwd(support_radius, mass, period, residuals, cotangents):
    """Run the backward twice, because the volume stands between the two forward passes.

    A particle's volume scales how every neighbor reads it, so that cotangent has to be
    complete before the positions can be resolved --- the same two-pass shape the
    forward has, for the same reason.
    """
    position, feature, density = residuals
    safe_density = jnp.maximum(density, 1e-8)
    volume = mass / safe_density

    _, feature_cotangent, volume_cotangent = _backward(
        position,
        feature,
        volume,
        cotangents,
        support_radius=support_radius,
        mass=mass,
        period=period,
        wants_position=False,
    )
    corrected = cotangents[0] - volume_cotangent * mass / jnp.square(safe_density)
    position_cotangent, _, _ = _backward(
        position,
        feature,
        volume,
        (corrected, *cotangents[1:]),
        support_radius=support_radius,
        mass=mass,
        period=period,
        wants_feature=False,
    )
    return position_cotangent, feature_cotangent


sph_moments.defvjp(_moments_fwd, _moments_bwd)
