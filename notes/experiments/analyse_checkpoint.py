"""Score every program of a cubff checkpoint with the replication detector and show the
most common tapes. Usage: analyse_checkpoint.py FILE.dat [--threshold 5]"""
import argparse, collections, numpy as np, jax, jax.numpy as jnp
from cax.cs.bff import replication_score, sample_partners, opcode_table_from_string, unparse, high_order_entropy

ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--threshold", type=int, default=5)
ap.add_argument("--npz", action="store_true")
a = ap.parse_args()
if a.npz:
    soup = np.load(a.file)["soup"]
else:
    raw = np.fromfile(a.file, dtype=np.uint8); n = int(raw[8:16].view(np.uint64)[0]); soup = raw[24:].reshape(n, 64)
print("programs", len(soup), "high-order entropy", round(high_order_entropy(soup), 3))
table = opcode_table_from_string()
scores = []
for i in range(0, len(soup), 2048):
    batch = jnp.asarray(soup[i:i + 2048])
    scores.append(np.asarray(replication_score(batch, sample_partners(jax.random.key(i), len(batch)), table)))
scores = np.concatenate(scores)
print("score histogram (score: count):", {int(k): int(v) for k, v in zip(*np.unique(scores, return_counts=True))})
print(f"replicators (score >= {a.threshold}): {(scores >= a.threshold).sum()}  (score >= 48): {(scores >= 48).sum()}")
common = collections.Counter(map(bytes, soup)).most_common(5)
for b, c in common:
    print(f"{c:5d}x  {unparse(np.frombuffer(b, dtype=np.uint8))}")
top = np.argsort(-scores)[:8]
for i in top:
    print(f"score {scores[i]:2d}: {unparse(soup[i])}")
