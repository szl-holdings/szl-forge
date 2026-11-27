"""Minimal safetensors reader (no torch, no safetensors package) for the LoRA no-op check."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from .canonical import ProofError

_SIGN_MASK = {"F64": (8, 0x7FFFFFFFFFFFFFFF), "F32": (4, 0x7FFFFFFF), "F16": (2, 0x7FFF),
              "BF16": (2, 0x7FFF)}


def read_header(path: Path | str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ProofError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ProofError(f"{path}: implausible safetensors header length {n}")
        try:
            hdr = json.loads(f.read(n))
        except Exception:
            raise ProofError(f"{path}: corrupt safetensors header")
    return hdr, 8 + n


def _nonzero_count(buf: bytes, dtype: str) -> int:
    """Number of elements that are not +/-0. Uses numpy when available."""
    width, mask = _SIGN_MASK.get(dtype, (1, 0xFF))
    try:
        import numpy as np
        ut = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        arr = np.frombuffer(buf, dtype=np.dtype(ut).newbyteorder("<"))
        return int(np.count_nonzero(arr & ut(mask)))
    except ImportError:  # pragma: no cover
        n = 0
        for i in range(0, len(buf), width):
            if int.from_bytes(buf[i:i + width], "little") & mask:
                n += 1
        return n


def lora_report(path: Path | str) -> dict:
    """Summarise LoRA tensors. no_op is True when there are no lora_B tensors or all are zero."""
    hdr, base = read_header(path)
    a_keys, b_keys, b_nonzero, b_elems = [], [], 0, 0
    with open(path, "rb") as f:
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            if "lora_A" in k or "lora_embedding_A" in k:
                a_keys.append(k)
            if "lora_B" in k or "lora_embedding_B" in k:
                b_keys.append(k)
                s, e = meta["data_offsets"]
                f.seek(base + s)
                buf = f.read(e - s)
                if len(buf) != e - s:
                    raise ProofError(f"{path}: truncated tensor {k}")
                width = _SIGN_MASK.get(meta["dtype"], (1, 0))[0]
                b_elems += len(buf) // width
                b_nonzero += _nonzero_count(buf, meta["dtype"])
    zero_b = b_nonzero == 0
    return {"lora_A_tensors": len(a_keys), "lora_B_tensors": len(b_keys),
            "lora_B_elements": b_elems, "lora_B_nonzero": b_nonzero,
            "no_op": (not b_keys) or zero_b}


def write_safetensors(path: Path | str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Test helper: tensors = {name: (dtype, shape, raw_bytes)}."""
    hdr, off, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
