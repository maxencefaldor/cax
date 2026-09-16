"""Per-step statistics: fraction alive, fraction taking a jump (needing a scan), on a random soup."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cax.cs.bff import initial_thread_state, opcode_table_from_string, step
from cax.cs.bff.language import Op

table = opcode_table_from_string()
rng = np.random.default_rng(3)
N = 4096
tapes = jnp.asarray(rng.integers(0, 256, (N, 128), dtype=np.uint8))


def need_of(state):
    idx = jnp.arange(N)
    cmd = table[state.tape[idx, state.pc]]
    v0 = state.tape[idx, state.head0]
    return (~state.halted) & (
        ((cmd == Op.LOOP_START) & (v0 == 0)) | ((cmd == Op.LOOP_END) & (v0 != 0))
    )


@jax.jit
def scan(state):
    def body(s, _):
        n = need_of(s)
        a = ~s.halted
        return jax.vmap(step, in_axes=(0, None))(s, table), (n.sum(), a.sum())

    return jax.lax.scan(body, state, None, length=8192)[1]


state = jax.vmap(partial(initial_thread_state, heads_from_tape=False))(tapes)
need, alive = map(np.asarray, scan(state))
for lo, hi in [
    (0, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 8192),
]:
    print(
        f"steps {lo:4d}-{hi:4d}: alive {alive[lo:hi].mean() / N:.3f}  need/N mean {need[lo:hi].mean() / N:.4f} max {need[lo:hi].max() / N:.4f}  need/alive {need[lo:hi].sum() / alive[lo:hi].sum():.3f}"
    )
