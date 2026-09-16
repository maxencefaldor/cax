"""Compare cax.cs.bff.replication_score with cubff's CheckSelfRep on identical noise."""
import os, subprocess, numpy as np, jax.numpy as jnp
from cax.cs.bff import replication_score, opcode_table_from_string, parse
H = os.environ.get("CUBFF_BIN", "cubff/bin") + "/harness_selfrep_"
table = opcode_table_from_string()
rng = np.random.default_rng(5)
ops = np.frombuffer(b"[]+-.,<>{}", dtype=np.uint8)
progs = rng.integers(0, 256, (200, 64), dtype=np.uint8)
m = rng.random(progs.shape); progs = np.where(m < 0.4, rng.choice(ops, progs.shape), progs).astype(np.uint8)
rep = np.zeros((4, 64), dtype=np.uint8)
for i, s in enumerate(["[[{.>]-]A]-]>.{[[", "[[{.>]-]]-]>.{[[", "[[{.>]-]]-]>.{[[" + "A"*20, "A"*10 + "[[{.>]-]A]-]>.{[["]):
    b = np.asarray(parse(s)); rep[i, :len(b)] = b
progs = np.concatenate([rep, progs])
for heads in (False, True):
    out = subprocess.run([H + ("bff" if heads else "noheads"), "42"], input=progs.tobytes(), capture_output=True, check=True).stdout
    rec = np.frombuffer(out, dtype=np.uint8).reshape(len(progs), 13 * 64 + 4)
    noise = rec[:, :13 * 64].reshape(len(progs), 13, 64).copy()
    ref = rec[:, 13 * 64:].copy().view(np.uint32).ravel()
    mine = np.asarray(replication_score(jnp.asarray(progs), jnp.asarray(noise), table, heads_from_tape=heads))
    print(f"heads={heads}: replicator scores ref={ref[:4].tolist()} mine={mine[:4].tolist()}; random progs: mismatches={int((ref != mine).sum())}/{len(progs)}, max ref score among random={int(ref[4:].max())}")
