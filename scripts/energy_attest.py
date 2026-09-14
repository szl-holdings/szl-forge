#!/usr/bin/env python3
"""
Energy attestation for gate runs — the first MEASURED energy in the SZL estate.

Everywhere else, energy reads UNAVAILABLE. This module samples NVML power draw
while a gate (or any callable) executes and integrates it into joules, emitting
a receipt with the same honesty discipline as the rest of the estate:

  - MEASURED only when NVML sampled the GPU throughout execution
  - UNAVAILABLE (never fabricated) when NVML is absent or sampling failed
  - receipts are unsigned-honest; the owner signs them into the chain

Usage (library):
    from energy_attest import EnergyAttest
    with EnergyAttest(gate="courier", model="SZLHOLDINGS/chaski-r2") as e:
        run_the_gate()
    print(e.receipt())

Usage (CLI wrapper — wraps any command):
    python energy_attest.py --gate courier --model SZLHOLDINGS/chaski-r2 \
        -- python scripts/run_heldout_gates.py --models chaski-r2
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "szl.energy-attest/v1"
SAMPLE_INTERVAL_S = 0.1


class EnergyAttest:
    def __init__(self, gate: str, model: str, label: str = "gpu"):
        self.gate = gate
        self.model = model
        self.label = label
        self.samples = []  # (monotonic_s, watts)
        self._handle = None
        self._nvml_ok = False

    def __enter__(self):
        try:
            import pynvml  # nvidia-ml-py
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # probe once to prove the path works before claiming MEASURED
            pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._nvml_ok = True
            self._t0 = time.monotonic()
            self._stop = False
            import threading
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._nvml_ok = False
        return self

    def _sample_loop(self):
        while not self._stop:
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append((time.monotonic(), mw / 1000.0))
            except Exception:
                self._nvml_ok = False
                return
            time.sleep(SAMPLE_INTERVAL_S)

    def __exit__(self, *exc):
        self._stop = True
        if self._nvml_ok:
            self._thread.join(timeout=2)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return False

    def _joules(self):
        if len(self.samples) < 2:
            return None
        total = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            total += ((w0 + w1) / 2.0) * (t1 - t0)
        return round(total, 2)

    def receipt(self) -> dict:
        j = self._joules()
        n = len(self.samples)
        watts = [w for _, w in self.samples]
        return {
            "schema": SCHEMA,
            "gate": self.gate,
            "model": self.model,
            "energy_status": "MEASURED" if (j is not None and self._nvml_ok) else "UNAVAILABLE",
            "energy_j": j,  # None when UNAVAILABLE — never fabricated
            "sample_count": n,
            "mean_power_w": round(sum(watts) / n, 2) if n else None,
            "peak_power_w": round(max(watts), 2) if n else None,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "device": "gpu:0" if self._nvml_ok else None,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "signature_state": "UNSIGNED-honest",
            "doctrine": "v11; energy never fabricated; UNAVAILABLE when not sampled",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER, help="command after '--'")
    args = ap.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command

    with EnergyAttest(gate=args.gate, model=args.model) as e:
        if cmd:
            proc = subprocess.run(cmd)
            rc = proc.returncode
        else:
            rc = 0

    receipt = e.receipt()
    out = Path("frontier/evaluation/energy")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{args.model.replace('/', '_')}_{args.gate}_{stamp}.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"receipt: {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
