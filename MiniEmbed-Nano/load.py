"""MiniEmbed-Nano loader.

Hash+table embed: little-endian SHA-256 token id modulo 64, mean-pool, L2.
The archive carries table[64,12] float64 + scalars V, D — this loader rebuilds
the documented forward pass against them. Not neural, not SVD, not the
3290x128 MiniEmbed on szl-kernels; this is the 64x12 hash table and nothing else.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np


def _l2(value, axis):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), 1e-12)
    except FloatingPointError as error:
        raise ValueError("MiniEmbed-Nano: non-finite normalization") from error


def _table(value):
    table = np.asarray(value)
    if table.shape != (64, 12) or table.dtype.kind not in "fiu":
        raise ValueError("MiniEmbed-Nano: expected a real table[64,12]")
    table = table.astype(np.float64)
    if not np.isfinite(table).all():
        raise ValueError("MiniEmbed-Nano: non-finite entry in table")
    return _l2(table, axis=1)


def load(path: str | pathlib.Path = "mini_embed.npz"):
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if ("table" not in names or not names <= {"table", "V", "D"}
                or len(names) != len(archive.files)):
            raise ValueError("MiniEmbed-Nano: expected table and optional V/D metadata")
        for name, expected in (("V", 64), ("D", 12)):
            if name in names:
                scalar = archive[name]
                if scalar.shape != () or scalar.dtype.kind not in "iu" or int(scalar) != expected:
                    raise ValueError(f"MiniEmbed-Nano: invalid {name} metadata")
        return _table(archive["table"]), 64, 12


def _row_id(token: str, v: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % v


def embed(text: str, table=None, path="mini_embed.npz"):
    """Canonical hash-table embedding; empty text uses normalized row zero.

    Matches szl_khipu/train/mini_embed.py at e438dd185aa1144c84d0ac81aee6d71c43bd0bf0.
    This fixture output is not a measured retrieval score or production model.
    """
    if not isinstance(text, str):
        raise ValueError("MiniEmbed-Nano: text must be a string")
    if table is None:
        table, v, _d = load(path)
    else:
        table = _table(table)
        v = 64
    tokens = text.split()
    if not tokens:
        return _l2(table[0].copy(), axis=0)
    rows = np.stack([table[_row_id(t, v)] for t in tokens])
    vec = rows.mean(axis=0)
    return _l2(vec, axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", default=[])
    parser.add_argument("--weights", default="mini_embed.npz")
    args = parser.parse_args()
    print(embed(" ".join(args.text) if args.text else "knot the run", path=args.weights))
