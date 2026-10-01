#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 5 - merge the adapter, convert to GGUF, quantize, and probe F16-vs-quant parity.
Writes out/merged/, out/gguf/*.gguf, out/QUANT_MANIFEST.json, out/parity_receipt.json, out/Modelfile.

Needs a llama.cpp checkout (--llama-cpp or env LLAMA_CPP) with convert_hf_to_gguf.py and a built
llama-quantize; the parity probe needs llama-simple or llama-cli. Anything missing is recorded UNAVAILABLE.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def find_bin(root, *names):
    for n in names:
        for pat in (n, n + ".exe"):
            hits = list(Path(root).rglob(pat))
            if hits:
                return str(hits[0])
    return lib.which(*names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    cand = lib.load_candidate()
    gates = lib.load_gates(cand)
    rep = lib.require_report("training_report.json", "state", ("TRAINED_CHALLENGER_UNSIGNED", "SMOKE_ONLY_NOT_A_CHALLENGER"))
    adapter = Path(args.adapter or rep["adapter_dir"])
    levels = list(cand["training_recipe"].get("export_levels") or ["Q4_K_M", "Q5_K_M", "Q8_0"])
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", cand["candidate_id"]).strip("-")
    merged, gguf_dir = lib.OUT / "merged", lib.OUT / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    import torch
    base = cand["actual_training_base"]
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base["repo_id"], revision=base.get("revision"),
                                                 dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = lib.attach_adapter(model, adapter).merge_and_unload()
    tok = lib.load_tokenizer(base["repo_id"], base.get("revision"))
    model.save_pretrained(str(merged), safe_serialization=True)
    tok.save_pretrained(str(merged))
    manifest = {"kind": "szl.quant-manifest/v1", "generated_utc": lib.now_utc(), "candidate_id": cand["candidate_id"],
                "parent": {"target_repo_id": cand["target_repo_id"], "adapterSha256": lib.sha256_dir(adapter),
                           "training_report_state": rep["state"]},
                "base": base, "merged_sha256": lib.sha256_dir(merged), "levels": {}, "converter": None,
                "status": "MERGED_ONLY"}
    parity = {"kind": "szl.quant-parity-receipt/v1", "generated_utc": lib.now_utc(), "verdict": "UNAVAILABLE",
              "agreement": None, "threshold": gates["quant_parity_min"], "detail": "no llama.cpp"}
    if args.llama_cpp and (Path(args.llama_cpp) / "convert_hf_to_gguf.py").is_file():
        conv = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
        code, out = run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"])
        manifest["converter"] = {"path": str(conv), "git_rev": out.strip() if code == 0 else None}
        f16 = gguf_dir / f"{slug}-F16.gguf"
        code, out = run([sys.executable, str(conv), str(merged), "--outtype", "f16", "--outfile", str(f16)])
        if code != 0 or not f16.is_file():
            manifest["status"] = "CONVERT_FAILED"
            manifest["convert_log_tail"] = out[-2000:]
        else:
            manifest["levels"]["F16"] = {"file": f16.name, "sha256": lib.sha256_file(f16), "bytes": f16.stat().st_size}
            quant = find_bin(args.llama_cpp, "llama-quantize")
            for lvl in levels:
                dst = gguf_dir / f"{slug}-{lvl}.gguf"
                if not quant:
                    manifest["levels"][lvl] = {"status": "NO_LLAMA_QUANTIZE"}
                    continue
                code, out = run([quant, str(f16), str(dst), lvl])
                manifest["levels"][lvl] = ({"file": dst.name, "sha256": lib.sha256_file(dst), "bytes": dst.stat().st_size}
                                           if code == 0 and dst.is_file() else {"status": "QUANTIZE_FAILED", "log_tail": out[-800:]})
            manifest["status"] = "QUANTIZED" if quant else "F16_ONLY"
            runner = find_bin(args.llama_cpp, "llama-simple", "llama-cli", "llama-completion")
            if runner:
                rows, _, _ = lib.load_splits(cand)
                probes = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
                          for r in (rows["dev"] or rows["train"])[:args.probes]]

                def decode(path):
                    outs = []
                    for p in probes:
                        code, out = run([runner, "-m", str(path), "-n", "48", "-p", p])
                        outs.append(re.sub(r"\s+", " ", out.split(p)[-1]).strip() if code == 0 else f"ERR{code}")
                    return outs
                ref = decode(f16)
                agree = {}
                for lvl, meta in manifest["levels"].items():
                    if lvl == "F16" or "file" not in meta:
                        continue
                    got = decode(gguf_dir / meta["file"])
                    agree[lvl] = sum(1 for a, b in zip(ref, got) if a == b) / max(1, len(ref))
                if agree:
                    worst = min(agree.values())
                    parity.update(agreement=agree, worst=worst, probes=len(probes), runner=runner,
                                  verdict="PASS" if worst >= float(gates["quant_parity_min"]) else "FAIL",
                                  detail="normalized greedy output equality per level vs F16")
            else:
                parity["detail"] = "no llama-simple/llama-cli binary found for the parity probe"
    else:
        manifest["status"] = "NO_LLAMA_CPP"
    q4 = manifest["levels"].get("Q4_K_M", {}).get("file")
    (lib.OUT / "Modelfile").write_text(
        f"FROM ./gguf/{q4 or slug + '-F16.gguf'}\nPARAMETER temperature 0\nPARAMETER num_ctx {cand['training_recipe']['max_length']}\n"
        f"# candidate {cand['candidate_id']} - not a release; see QUANT_MANIFEST.json and parity_receipt.json\n", encoding="utf-8")
    lib.write_json(lib.OUT / "QUANT_MANIFEST.json", lib.stringify_floats(manifest))
    lib.write_json(lib.OUT / "parity_receipt.json", lib.stringify_floats(parity))
    print(f"[export] {manifest['status']} levels={list(manifest['levels'])} parity={parity['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
