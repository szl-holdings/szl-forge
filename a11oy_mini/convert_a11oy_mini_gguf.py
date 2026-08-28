#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""A11OY-MINI GGUF convert — llama.cpp rebirth of LIVE Chaski, Python not PowerShell.

Parent: SZLHOLDINGS/chaski merged shard. NOT SZLHOLDINGS/chaski-5050.
Not a third LLM. Not a new train. GGUF SKU of live Chaski.

House path (same as rebirth.ps1 / rebirth-khipu.ps1, in Python):
  1) F16 GGUF via llama.cpp convert_hf_to_gguf.py --outtype f16
  2) Q4_K_M via llama-quantize (or ollama create FROM the F16 GGUF)

BANNED (MEASURED 2026-07-12 @ spam): direct safetensors → ollama create.
This script MUST NOT Hub PUT. Scripts only this week. No GGUF bytes in this PR.
FORGE does not push after this PR exists.
Hub GGUF only after live Chaski is publication-eligible (INTI).
publication_eligible false. No base_model_relation quantized.
Do not overwrite SZLHOLDINGS/chaski. Do not pin the Khipu lab.
Do not claim tok/s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

SKU = "SZLHOLDINGS/A11OY-MINI"
PARENT = "SZLHOLDINGS/chaski"
FORBIDDEN_PARENT = "SZLHOLDINGS/chaski-5050"
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
MERGED_SHARD_NAME = "model.safetensors-00001-of-00001.safetensors"
SEED = 11
DOCTRINE = "v11 LOCKED 749/14/163"
F16_NAME = "a11oy-mini-f16.gguf"
Q4_NAME = "a11oy-mini-q4_k_m.gguf"
QUANT = "Q4_K_M"
LLAMA_CPP_CONVERT = "convert_hf_to_gguf.py"
LLAMA_CPP_CLONE = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_TARBALL = (
    "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz"
)
KHIPU_LAB = "SZLHOLDINGS/szl-model-inference-lab"
KHIPU_GGUF = "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF"

UPLOAD_FLAG_NAMES = (
    "--upload",
    "--push",
    "--push-to-hub",
    "--hub-put",
    "--publish",
)
FORBIDDEN_PARENT_MARKERS = (
    "chaski-5050",
    "SZLHOLDINGS/chaski-5050",
    "train_chaski_bf16_5050",
    "local-5050",
)


class ConvertError(SystemExit):
    """Fail closed. Never a Hub PUT."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_5050(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in FORBIDDEN_PARENT_MARKERS)


def refuse_hub_put(*flags: str) -> None:
    """GitHub kit only. Scripts only this week. No Hub PUT. No GGUF bytes.

    FORGE does not push after this PR exists.
    Hub GGUF only after live Chaski is publication-eligible (INTI).
    """
    offered = [flag for flag in flags if flag in UPLOAD_FLAG_NAMES]
    if offered:
        raise ConvertError(
            "[a11oy-mini] refusing Hub PUT from this checkout "
            f"({', '.join(offered)}). Scripts only this week. No GGUF bytes. "
            "FORGE does not push after this PR exists. Hub GGUF only after live "
            "Chaski is publication-eligible (INTI). "
            f"Do not upload GGUF or an empty parent to {SKU}."
        )


def refuse_5050_parent(source: Path | str | None) -> None:
    if source is None:
        return
    blob = str(source)
    if _looks_like_5050(blob):
        raise ConvertError(
            "[a11oy-mini] refusing 5050 parent. "
            f"Parent is live {PARENT} merged shard, not {FORBIDDEN_PARENT}."
        )
    path = Path(source)
    if path.is_dir():
        for name in ("README.md", "adapter_config.json", "training_receipt.json"):
            candidate = path / name
            if candidate.is_file() and _looks_like_5050(
                candidate.read_text(encoding="utf-8", errors="replace")
            ):
                raise ConvertError(
                    "[a11oy-mini] refusing 5050 parent at "
                    f"{candidate}. Parent is live {PARENT}."
                )


def refuse_live_chaski_overwrite(target: str | None) -> None:
    if not target:
        return
    normalized = target.strip().rstrip("/")
    if normalized in {PARENT, f"https://huggingface.co/{PARENT}"}:
        raise ConvertError(
            f"[a11oy-mini] refusing overwrite of live {PARENT}. "
            f"SKU pin is {SKU}."
        )


def refuse_safetensors_ollama(from_spec: str | Path) -> None:
    """MEASURED 2026-07-12: Ollama direct safetensors import produced @ spam."""
    spec = Path(str(from_spec).replace("FROM", "").strip())
    if spec.suffix.lower() == ".gguf":
        return
    if spec.is_dir():
        tensors = list(spec.glob("*.safetensors")) + list(
            spec.glob("**/*.safetensors")
        )
        if tensors:
            raise ConvertError(
                "[a11oy-mini] BANNED: direct safetensors→Ollama "
                f"(MEASURED 2026-07-12 @ spam). Saw {tensors[0].name}. "
                "Convert F16 GGUF first, then Q4_K_M from that GGUF."
            )
    if spec.suffix.lower() in {".safetensors", ".bin"}:
        raise ConvertError(
            "[a11oy-mini] BANNED: direct safetensors→Ollama "
            "(MEASURED 2026-07-12 @ spam). llama.cpp F16 then Q4_K_M only."
        )


def find_merged_dir(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("A11OY_MINI_MERGED")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            HERE / "chaski-merged",
            REPO_ROOT / "chaski" / "chaski-merged",
            REPO_ROOT / "chaski-merged",
        ]
    )
    for path in candidates:
        refuse_5050_parent(path)
        shard = path / MERGED_SHARD_NAME
        if path.is_dir() and shard.is_file():
            return path
        # A merged HF export may use model.safetensors instead of the Hub shard name.
        if path.is_dir() and any(path.glob("*.safetensors")):
            if (path / "config.json").is_file():
                return path
    return None


def find_llama_converter(llama_root: Path | None = None) -> Path | None:
    roots = []
    if llama_root is not None:
        roots.append(llama_root)
    env = os.environ.get("A11OY_MINI_LLAMA_CPP")
    if env:
        roots.append(Path(env))
    roots.extend([HERE / "llama.cpp", REPO_ROOT / "llama.cpp"])
    for root in roots:
        converter = root / LLAMA_CPP_CONVERT
        if converter.is_file():
            return converter
    return None


def ensure_llama_converter(*, fetch: bool = False) -> Path:
    existing = find_llama_converter()
    if existing is not None:
        return existing
    if not fetch:
        raise ConvertError(
            "[a11oy-mini] llama.cpp converter missing. "
            f"Need {LLAMA_CPP_CONVERT}. Pass --fetch-llama-cpp or set "
            "A11OY_MINI_LLAMA_CPP."
        )
    dest = HERE / "llama.cpp"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_CLONE, str(dest)],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        converter = dest / LLAMA_CPP_CONVERT
        if completed.returncode == 0 and converter.is_file():
            return converter
    archive = HERE / "llama-cpp.tar.gz"
    urllib.request.urlretrieve(LLAMA_CPP_TARBALL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(HERE)
    extracted = next(HERE.glob("llama.cpp-*"), None)
    if extracted is not None and extracted.is_dir():
        extracted.rename(dest)
    converter = dest / LLAMA_CPP_CONVERT
    if not converter.is_file():
        raise ConvertError("[a11oy-mini] could not obtain llama.cpp converter")
    return converter


def find_llama_quantize(llama_root: Path | None = None) -> Path | None:
    names = ("llama-quantize", "quantize")
    roots: list[Path] = []
    if llama_root is not None:
        roots.append(llama_root)
    env = os.environ.get("A11OY_MINI_LLAMA_CPP")
    if env:
        roots.append(Path(env))
    roots.extend([HERE / "llama.cpp", REPO_ROOT / "llama.cpp"])
    which = shutil.which("llama-quantize")
    if which:
        return Path(which)
    for root in roots:
        for name in names:
            for candidate in (
                root / name,
                root / "build" / "bin" / name,
                root / "build" / name,
            ):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    return None


def gguf_bytes_label(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "path": None,
            "bytes": None,
            "sha256": None,
            "label": "ROADMAP",
        }
    digest = sha256_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "label": "MEASURED",
    }


def conversion_receipt(
    *,
    merged: Path | None,
    f16_path: Path | None,
    q4_path: Path | None,
    live: bool = False,
) -> dict[str, Any]:
    f16 = gguf_bytes_label(f16_path)
    q4 = gguf_bytes_label(q4_path)
    gguf_exists = bool(f16_path and f16_path.is_file()) or bool(
        q4_path and q4_path.is_file()
    )
    return {
        "kind": "szl-a11oy-mini-conversion-receipt",
        "schema": "szl.gguf-conversion-run/v1",
        "artifact": SKU,
        "parent": PARENT,
        "forbidden_parent": FORBIDDEN_PARENT,
        "parent_artifact": "merged-shard",
        "parent_merged_shard": MERGED_SHARD_NAME,
        "base_model": CANONICAL_BASE,
        "silhouette": "Qwen3.5 instruct",
        "cut": "live SZL Chaski",
        "sku_class": "GGUF_SKU_OF_LIVE_CHASKI",
        "third_llm": False,
        "new_train": False,
        "seed": SEED,
        "doctrine": DOCTRINE,
        "lambda": "Conjecture 1",
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "evals": "none-this-run",
        "quality": "ROADMAP",
        "card_status": "ROADMAP",
        "bytes_measured": bool(f16["sha256"] or q4["sha256"]),
        "gguf_exists": gguf_exists,
        "base_model_relation_quantized": False,
        "hub_put": False,
        "khipu_lab_pin": False,
        "inference_lab_pin": False,
        "khipu_lab": KHIPU_LAB,
        "khipu_gguf": KHIPU_GGUF,
        "tok_s_claim": False,
        "direct_safetensors_ollama": "BANNED",
        "convert_path": [
            "llama.cpp convert_hf_to_gguf.py --outtype f16",
            f"llama-quantize {QUANT}",
        ],
        "merged_dir": str(merged) if merged else None,
        "f16": f16,
        "q4_k_m": q4,
        "claim_boundary": (
            "A11OY-MINI is a later GGUF SKU of live SZLHOLDINGS/chaski, "
            "not a new train and not chaski-5050. Evals inherit "
            "none-this-run. publication_eligible false. Quality ROADMAP. "
            "No base_model_relation quantized. Direct safetensors→Ollama banned. "
            "Scripts only this week. No GGUF bytes. "
            "FORGE does not push after this PR exists. Hub GGUF only after live "
            "Chaski is publication-eligible (INTI). "
            "No Hub PUT from this checkout. Lab stays Khipu. No tok/s."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "forge-status" if not live else "local-convert",
    }


def write_receipt(receipt: dict[str, Any], path: Path | None = None) -> Path:
    dest = path or (HERE / "conversion_receipt.json")
    dest.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return dest


def convert_f16(
    merged: Path,
    outfile: Path,
    converter: Path,
    *,
    python_bin: str | None = None,
) -> Path:
    refuse_5050_parent(merged)
    cmd = [
        python_bin or sys.executable,
        str(converter),
        str(merged),
        "--outfile",
        str(outfile),
        "--outtype",
        "f16",
    ]
    print(f"[a11oy-mini] F16: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=HERE)
    if completed.returncode != 0 or not outfile.is_file():
        raise ConvertError("[a11oy-mini] F16 GGUF conversion failed")
    return outfile


def write_f16_modelfile(f16_path: Path, dest: Path) -> Path:
    refuse_safetensors_ollama(f16_path)
    dest.write_text(
        f"FROM {f16_path}\n"
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    return dest


def quantize_q4_k_m(f16_path: Path, outfile: Path) -> Path:
    refuse_safetensors_ollama(f16_path)
    if not f16_path.is_file() or f16_path.suffix.lower() != ".gguf":
        raise ConvertError(
            "[a11oy-mini] Q4_K_M requires a local F16 GGUF. "
            "Direct safetensors→Ollama is BANNED."
        )
    quantize = find_llama_quantize()
    if quantize is not None:
        cmd = [str(quantize), str(f16_path), str(outfile), QUANT]
        print(f"[a11oy-mini] Q4_K_M: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=HERE)
        if completed.returncode != 0 or not outfile.is_file():
            raise ConvertError("[a11oy-mini] llama-quantize Q4_K_M failed")
        return outfile
    ollama = shutil.which("ollama")
    if ollama:
        with tempfile.TemporaryDirectory(prefix="a11oy-mini-modelfile-") as tmp:
            modelfile = Path(tmp) / "Modelfile"
            write_f16_modelfile(f16_path, modelfile)
            # Quantize FROM the F16 GGUF. Never FROM a safetensors directory.
            cmd = [
                ollama,
                "create",
                "a11oy-mini-local",
                "--quantize",
                "q4_K_M",
                "-f",
                str(modelfile),
            ]
            print(
                "[a11oy-mini] Q4_K_M via ollama create FROM F16 GGUF "
                "(not safetensors)"
            )
            completed = subprocess.run(cmd, cwd=HERE)
            if completed.returncode != 0:
                raise ConvertError(
                    "[a11oy-mini] ollama Q4_K_M from F16 GGUF failed"
                )
        if outfile.is_file():
            return outfile
        raise ConvertError(
            "[a11oy-mini] ollama quantized from F16 GGUF but did not write "
            f"{outfile.name}. Export the Q4_K_M blob locally; no Hub PUT."
        )
    raise ConvertError(
        "[a11oy-mini] Q4_K_M needs llama-quantize or ollama create FROM the "
        "F16 GGUF. Direct safetensors→Ollama is BANNED."
    )


def status_main() -> int:
    merged = find_merged_dir()
    f16 = HERE / F16_NAME
    q4 = HERE / Q4_NAME
    receipt = conversion_receipt(
        merged=merged,
        f16_path=f16 if f16.is_file() else None,
        q4_path=q4 if q4.is_file() else None,
        live=False,
    )
    path = write_receipt(receipt)
    print(f"[a11oy-mini] sku={SKU} parent={PARENT} not={FORBIDDEN_PARENT}")
    print(f"[a11oy-mini] base={CANONICAL_BASE} seed={SEED} doctrine=v11")
    print(
        "[a11oy-mini] status=ROADMAP evals=none-this-run "
        "publication_eligible=false hub_put=false"
    )
    print(
        f"[a11oy-mini] gguf_exists={receipt['gguf_exists']} "
        f"bytes_measured={receipt['bytes_measured']} "
        "lab=Khipu tok_s=false"
    )
    print(f"[a11oy-mini] wrote {path}")
    return 0


def convert_main(args: argparse.Namespace) -> int:
    refuse_hub_put(*getattr(args, "unknown", ()))
    refuse_live_chaski_overwrite(getattr(args, "hub_id", None))
    merged = find_merged_dir(Path(args.merged) if args.merged else None)
    if merged is None:
        raise ConvertError(
            "[a11oy-mini] no live Chaski merged shard. Place the Hub merge "
            f"({MERGED_SHARD_NAME} from {PARENT}) at a11oy_mini/chaski-merged "
            "or pass --merged. Not 5050. ROADMAP until a .gguf exists."
        )
    refuse_5050_parent(merged)
    converter = ensure_llama_converter(fetch=args.fetch_llama_cpp)
    f16_path = Path(args.f16_out) if args.f16_out else HERE / F16_NAME
    q4_path = Path(args.q4_out) if args.q4_out else HERE / Q4_NAME
    convert_f16(merged, f16_path, converter)
    quantize_q4_k_m(f16_path, q4_path)
    receipt = conversion_receipt(
        merged=merged,
        f16_path=f16_path,
        q4_path=q4_path,
        live=True,
    )
    path = write_receipt(receipt)
    print(f"[a11oy-mini] F16 {f16_path} sha256={receipt['f16']['sha256']}")
    print(f"[a11oy-mini] {QUANT} {q4_path} sha256={receipt['q4_k_m']['sha256']}")
    print(f"[a11oy-mini] wrote {path} hub_put=false")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    for flag in raw:
        token = flag.split("=", 1)[0]
        if token in UPLOAD_FLAG_NAMES:
            refuse_hub_put(token)
        if "huggingface.co" in flag.lower() and any(
            verb in flag.lower() for verb in ("upload", "put", "push")
        ):
            raise ConvertError(
                "[a11oy-mini] refusing Hub PUT from this checkout."
            )
        parsed = urlparse(flag)
        if parsed.scheme in {"http", "https"} and "huggingface.co" in parsed.netloc:
            # Read-only Hub GET of live Chaski is allowed; PUT is not an argv URL we issue.
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Run llama.cpp F16 then Q4_K_M from the live Chaski merge",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Write an honest ROADMAP receipt without converting",
    )
    parser.add_argument("--merged", help="Path to live Chaski merged HF dir")
    parser.add_argument("--f16-out", help=f"F16 outfile (default {F16_NAME})")
    parser.add_argument("--q4-out", help=f"Q4_K_M outfile (default {Q4_NAME})")
    parser.add_argument(
        "--fetch-llama-cpp",
        action="store_true",
        help="Clone/download llama.cpp converter if missing",
    )
    parser.add_argument(
        "--hub-id",
        default=SKU,
        help="Must stay SZLHOLDINGS/A11OY-MINI; live Chaski overwrite refused",
    )
    args = parser.parse_args(raw)
    refuse_live_chaski_overwrite(args.hub_id)
    if args.hub_id not in {SKU, "A11OY-MINI"}:
        raise ConvertError(
            f"[a11oy-mini] serve/convert pin is {SKU}, not {args.hub_id}"
        )
    if args.convert:
        return convert_main(args)
    return status_main()


if __name__ == "__main__":
    raise SystemExit(main())
