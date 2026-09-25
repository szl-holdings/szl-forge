"""Run a bounded, untrained CPU continuity experiment and export real evidence."""
from __future__ import annotations

import argparse
from dataclasses import replace
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch

from .model import TinyRLT
from .runtime import ContinuityError, Scope, Session, canonical, digest


def run() -> tuple[dict[str, object], bytes]:
    """Deterministic synthetic tokens; measured continuity, not language quality."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = TinyRLT().eval()
    scope = Scope(**{name: digest(canonical({"synthetic_scope": name})) for name in Scope.__dataclass_fields__})
    start = time.perf_counter()
    session = Session(model, scope)
    session.append([1, 2, 3, 4])
    pending = session.emit_greedy()
    artifact = session.checkpoint()
    restored = Session.restore(model, scope, artifact, expected_sha256=digest(artifact),
                               consumed_tokens=[1, 2, 3, 4], pending_token=pending)
    expected, actual = session.append([pending, 5, 6]), restored.append([pending, 5, 6])
    error = float((expected - actual).abs().max())
    checks = {"checkpoint_restore_exact": bool(torch.equal(expected, actual)),
              "consumed_once": restored.consumed == 7}
    changed = replace(scope, evidence_sha256=digest(b"new-synthetic-evidence"))
    try:
        Session.restore(model, changed, artifact, expected_sha256=digest(artifact),
                        consumed_tokens=[1, 2, 3, 4], pending_token=pending)
    except ContinuityError:
        checks["stale_evidence_rejected"] = True
    else:
        checks["stale_evidence_rejected"] = False
    replayed = session.replay(changed, [1, 2, 3, 4])
    checks["fresh_evidence_replay"] = replayed.observation()["identity_sha256"] != session.observation()["identity_sha256"]
    report = {"schema": "szl.rlt.cpu-continuity-evaluation.v1", "scope": "UNTRAINED_SYNTHETIC_REFERENCE",
              "status": "MEASURED_LOCAL_PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "max_abs_restore_error": error, "checkpoint_sha256": digest(artifact),
              "checkpoint_bytes": len(artifact), "parameter_count": sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.perf_counter() - start,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "safetensors": importlib.metadata.version("safetensors"), "device": "cpu"},
              "source_sha256": {name: digest(Path(__file__).with_name(name).read_bytes())
                                for name in ("model.py", "runtime.py", "cli.py")},
              "observation": restored.observation(), "trained": False, "production_qualified": False,
              "execution_authority": "NONE", "receipt_status": "UNSIGNED",
              "limits": ["No language-quality or speedup evaluation", "No GPU qualification",
                         "No HF publication or domain deployment", "Digest integrity is not producer authentication",
                         "Checkpoint is private inference state, not public proof or a training checkpoint"]}
    report["receipt_sha256"] = digest(canonical(report))
    return report, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new local directory; existing paths are refused")
    args = parser.parse_args()
    # No network, provider credentials, existing file overwrite, or external code.
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    report, artifact = run()
    (args.output / "checkpoint.safetensors").write_bytes(artifact)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "receipt.json").write_text(rendered, encoding="utf-8")
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>SZL recurrent continuity evidence</title><style>'
            'body{font:1rem/1.6 system-ui;max-width:70rem;margin:2rem auto;padding:0 1rem}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>Recurrent continuity — measured CPU reference</h1>'
            '<p>Untrained synthetic execution. No production qualification or execution authority.</p>'
            '<h2>Actual run receipt</h2><pre>' + html.escape(rendered) + '</pre></html>')
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "MEASURED_LOCAL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
