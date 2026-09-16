"""Replay cubff's RNG (SplitMix64 init, Fisher-Yates shuffle, per-byte mutation) host-side and
check that CAX's pair_and_run reproduces cubff's per-epoch checkpoints byte for byte."""

import os

import jax.numpy as jnp
import numpy as np
from flax import nnx

from cax.cs.bff import BFF

M = (1 << 64) - 1


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def replay(lang_dir, heads, params_seed=7, N=1024, epochs=4, mutation_prob=1 << 18):
    seed = lambda s2: splitmix64(splitmix64(params_seed) ^ splitmix64(s2))
    # InitPrograms
    s0 = seed(0)
    soup = np.array(
        [
            [splitmix64((64 * N * s0 + 64 * i + j) & M) % 256 for j in range(64)]
            for i in range(N)
        ],
        dtype=np.uint8,
    )
    cs = BFF(mutation_rate=0.0, heads_from_tape=heads, rngs=nnx.Rngs(0))
    ok = True
    for epoch in range(epochs):
        # do_shuffle: for i = N-1 .. 0: j = SplitMix64(seed(epoch*N + i)) % (i+1); swap
        perm = np.arange(N, dtype=np.int64)
        for i in range(N - 1, -1, -1):
            j = splitmix64(seed(epoch * N + i)) % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        # mutation, applied to the concatenated pair tapes
        se = seed(epoch)
        pairs = soup[perm].reshape(N // 2, 128).copy()
        for index in range(N // 2):
            for i in range(128):
                r = splitmix64(((N * se + index) * 128 + i) & M)
                if ((r >> 8) & ((1 << 30) - 1)) < mutation_prob:
                    pairs[index, i] = r & 0xFF
        mutated = np.zeros_like(soup)
        mutated[perm] = pairs.reshape(N, 64)
        out, _ = cs.pair_and_run(jnp.asarray(mutated), jnp.asarray(perm))
        soup = np.asarray(out)
        ref = np.fromfile(f"{lang_dir}/{epoch:010d}.dat", dtype=np.uint8)[24:].reshape(
            N, 64
        )
        diff = int((soup != ref).sum())
        print(f"{lang_dir} epoch {epoch}: mismatching bytes {diff} / {N * 64}")
        ok &= diff == 0
        soup = ref  # continue from the reference either way
    return ok


R = os.environ.get("CUBFF_REPLAY", "replay") + "/"
a = replay(R + "nh1024", heads=False)
b = replay(R + "h1024", heads=True)
print("REPLAY EXACT" if a and b else "REPLAY MISMATCH")
