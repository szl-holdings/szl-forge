"""MiniEmbed-Nano loader.

Hash+table embed: SHA-256 token id modulo 64, mean-pool the rows, L2-normalize.
The archive carries table[64,12] float64 + scalars V, D — this loader rebuilds
the documented forward pass against them. Not neural, not SVD, not the
3290x128 MiniEmbed on szl-kernels; this is the 64x12 hash table and nothing else.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np


def load(path: str | pathlib.Path = "mini_embed.npz"):
    z = np.load(path, allow_pickle=False)
    if "table" not in z.files:
        raise ValueError(f"MiniEmbed-Nano: missing 'table'; have {sorted(z.files)}")
    table = z["table"].astype(np.float64)
    v = int(z["V"]) if "V" in z.files else table.shape[0]
    d = int(z["D"]) if "D" in z.files else table.shape[1]
    if table.shape != (v, d):
        raise ValueError(f"MiniEmbed-Nano: table {table.shape} != ({v}, {d})")
    if not np.isfinite(table).all():
        raise ValueError("MiniEmbed-Nano: non-finite entry in table")
    return table, v, d


def _row_id(token: str, v: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % v


def embed(text: str, table=None, path="mini_embed.npz"):
    """text -> L2-normalized d-vector. Empty text returns the zero vector."""
    if table is None:
        table, v, _d = load(path)
    else:
        v = table.shape[0]
    tokens = text.split()
    if not tokens:
        return np.zeros(table.shape[1])
    rows = np.stack([table[_row_id(t, v)] for t in tokens])
    vec = rows.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "knot the run"
    print(embed(text))
