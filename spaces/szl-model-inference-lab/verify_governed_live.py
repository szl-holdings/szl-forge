#!/usr/bin/env python3
"""Verify the deployed governed inference surface without retaining prompt/output text."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT = "Answer with exactly this status, then cite evidence: Lambda remains Conjecture 1, advisory only."
FORGE_CONTROLLER_REVISION = "943f6ab987bbe120cae32649c46c3a5f0b6f9e9b"
SECOND_BRAIN_REVISION = "fa3e4605344b13db220a79f9dcd267ee5725c87e"
NEMO_REVISION = "810231a531188bb569e3faa17396386eb0a5e260"
MODEL_REVISION = "67d60ec577730747055491640cfb91fc4a4b5d25"
LOCKED_EIGHT = ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"]
BANNED_PERSISTED_KEYS = {
    "prompt",
    "raw_prompt",
    "content",
    "raw_content",
    "hydrated_content",
    "chain_of_thought",
    "private_chain_of_thought",
    "hidden_reasoning",
    "reasoning_trace",
    "raw_private_graph",
    "private_graph",
}


class VerificationError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 90.0,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = canonical_bytes(payload) if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise VerificationError(f"{url} did not return a JSON object")
            return (
                int(response.status),
                body,
                {key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"detail": f"HTTP {exc.code}"}
        if not isinstance(body, dict):
            body = {"detail": f"HTTP {exc.code}"}
        return int(exc.code), body, {
            key.lower(): value for key, value in exc.headers.items()
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def walk_banned_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child = f"{path}.{key_text}"
            if key_text.lower() in BANNED_PERSISTED_KEYS:
                found.append(child)
            found.extend(walk_banned_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(walk_banned_keys(item, f"{path}[{index}]"))
    return found


def wait_for_ready(
    base_url: str,
    *,
    expected_source_revision: str,
    attempts: int = 24,
    delay_seconds: float = 15.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    last: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        status, body, headers = request_json(
            f"{base_url}/api/v2/governed-health", timeout=30.0
        )
        last = body
        if (
            status == 200
            and body.get("status") == "READY"
            and body.get("source_revision") == expected_source_revision
        ):
            return body, headers
        if attempt != attempts:
            time.sleep(delay_seconds)
    raise VerificationError(
        "governed health did not become READY at the expected source revision; "
        f"last_state={last.get('status')} last_source={last.get('source_revision')}"
    )


def verify_health(
    health: dict[str, Any],
    headers: dict[str, str],
    expected_source_revision: str,
) -> None:
    require(health.get("status") == "READY", "governed health is not READY")
    require(
        health.get("source_revision") == expected_source_revision,
        "deployed source revision mismatch",
    )
    require(
        health.get("controller_revision") == FORGE_CONTROLLER_REVISION,
        "Forge controller revision mismatch",
    )
    brain = health.get("second_brain") or {}
    require(brain.get("ready") is True, "Second Brain is not ready")
    require(brain.get("public_chunk_count") == 575, "Second Brain count drift")
    require(
        brain.get("content_access") == "HANDLES_ONLY",
        "Second Brain public boundary drift",
    )
    require(
        brain.get("private_graph_present") is False,
        "private graph appeared in public runtime",
    )
    nemo = health.get("nemo") or {}
    require(nemo.get("version") == "0.4.0", "Nemo version drift")
    require(
        nemo.get("envelope_rules") == "doctrine-v11/E1-E10",
        "Nemo envelope rules drift",
    )
    require(
        nemo.get("text_rules") == "doctrine-v11/R1-R5",
        "Nemo text rules drift",
    )
    packages = (health.get("dependency_status") or {}).get("packages") or {}
    for name, expected in (
        ("szl-forge-inference", "0.2.0"),
        ("szl-second-brain", "1.2.0"),
        ("szl-nemo", "0.4.0"),
    ):
        observed = packages.get(name) or {}
        require(observed.get("match") is True, f"{name} package mismatch")
        require(observed.get("observed") == expected, f"{name} version mismatch")
    require(
        headers.get("x-szl-forge-controller-revision")
        == FORGE_CONTROLLER_REVISION,
        "Forge controller header mismatch",
    )
    require(
        headers.get("x-szl-second-brain-revision") == SECOND_BRAIN_REVISION,
        "Second Brain header mismatch",
    )
    require(
        headers.get("x-szl-nemo-revision") == NEMO_REVISION,
        "Nemo header mismatch",
    )


def verify_contract(contract: dict[str, Any]) -> None:
    endpoint = contract.get("endpoint") or {}
    require(
        endpoint.get("path") == "/api/v2/governed-infer",
        "governed endpoint path drift",
    )
    require(endpoint.get("tools") is False, "public tools must remain disabled")
    controller = contract.get("controller") or {}
    require(
        controller.get("revision") == FORGE_CONTROLLER_REVISION,
        "contract controller revision mismatch",
    )
    brain = contract.get("second_brain") or {}
    require(
        brain.get("revision") == SECOND_BRAIN_REVISION,
        "contract Second Brain revision mismatch",
    )
    require(
        brain.get("private_graph_present") is False,
        "contract exposes private graph",
    )
    nemo = contract.get("nemo") or {}
    require(nemo.get("revision") == NEMO_REVISION, "contract Nemo revision mismatch")
    require(
        nemo.get("structured_witness") == "doctrine-v11/E1-E10",
        "contract Nemo structured witness drift",
    )
    formula = contract.get("formula_authority") or {}
    require(
        formula.get("locked_proven_count") == 8,
        "locked formula count drift",
    )
    require(
        formula.get("locked_proven_ids") == LOCKED_EIGHT,
        "locked formula identity drift",
    )
    lambda_rule = formula.get("lambda") or {}
    require(
        lambda_rule.get("status") == "CONJECTURE_1_ADVISORY",
        "Lambda status drift",
    )
    require(
        lambda_rule.get("can_authorize") is False,
        "Lambda gained action authority",
    )
    runtime = contract.get("runtime_selection") or {}
    require(runtime.get("winner") == "UNSELECTED", "runtime winner fabricated")
    model = contract.get("model") or {}
    require(model.get("revision") == MODEL_REVISION, "model revision drift")


def verify_inference(
    result: dict[str, Any],
    headers: dict[str, str],
    expected_source_revision: str,
) -> dict[str, Any]:
    require(result.get("state") == "PROPOSAL", "live inference is not PROPOSAL")
    require(result.get("executed") is False, "Forge executed a tool")
    require(
        result.get("authority_state") == "NO_ACTION_AUTHORITY",
        "public model gained action authority",
    )
    output = str(result.get("output") or "")
    require(bool(output), "live inference returned no output")
    require(
        result.get("output_sha256") == text_sha256(output),
        "output digest mismatch",
    )
    require(bool(result.get("evidence_handles")), "no evidence handles returned")
    require(bool(result.get("claims")), "no claim receipts returned")
    require(
        result.get("claims_sha256") == canonical_sha256(result.get("claims")),
        "claim-set digest mismatch",
    )
    require(
        result.get("citations_sha256") == canonical_sha256(result.get("citations")),
        "citation-set digest mismatch",
    )
    model = result.get("model") or {}
    require(model.get("revision") == MODEL_REVISION, "result model revision drift")
    require(
        model.get("template_revision") == expected_source_revision,
        "template/source revision binding mismatch",
    )
    require(
        model.get("quantization_revision")
        == "sha256:13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a",
        "GGUF quantization digest drift",
    )
    nemo = result.get("nemo") or []
    require(
        [(item.get("stage"), item.get("decision")) for item in nemo]
        == [("PRE_GENERATION", "ALLOW"), ("POST_GENERATION", "ALLOW")],
        "Nemo E1-E10 witness sequence failed",
    )
    text_witness = (result.get("metrics") or {}).get("nemo_text_witness") or {}
    require(
        text_witness.get("decision") == "ALLOW",
        "Nemo R1-R5 text witness failed",
    )
    require(
        text_witness.get("rule_version") == "doctrine-v11/R1-R5",
        "Nemo text rule version drift",
    )
    receipt = result.get("receipt") or {}
    require(
        receipt.get("schema") == "szl.forge.production-inference-receipt/v2",
        "receipt schema drift",
    )
    require(
        receipt.get("receipt_sha256") == canonical_sha256(receipt.get("payload")),
        "receipt digest mismatch",
    )
    signature = receipt.get("signature") or {}
    require(
        signature.get("status") == "UNSIGNED_LOCAL",
        "unsigned inference receipt mislabeled",
    )
    require(
        signature.get("must_be_signed_before_consequential_action") is True,
        "action signing boundary drift",
    )
    anatomy = result.get("anatomy_observation") or {}
    require(anatomy.get("delivery") == "DELIVERED", "Anatomy event not delivered")
    event = anatomy.get("event") or {}
    require(event.get("observer_authority") == "NONE", "Anatomy gained authority")
    for key in (
        "raw_prompt_present",
        "hydrated_content_present",
        "private_reasoning_present",
    ):
        require(event.get(key) is False, f"unsafe Anatomy flag: {key}")
    require(PROMPT not in json.dumps(result, sort_keys=True), "raw prompt persisted")
    banned = walk_banned_keys(result)
    require(not banned, f"forbidden persisted fields: {banned}")
    require(
        headers.get("x-szl-governed-inference") == "v2",
        "governed response header missing",
    )
    return {
        "output_sha256": result["output_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "claims_sha256": result["claims_sha256"],
        "citations_sha256": result["citations_sha256"],
        "evidence_set_sha256": result["evidence_set_sha256"],
        "nemo_input_hashes": [item.get("input_hash") for item in nemo],
        "text_witness_input_hash": text_witness.get("input_hash"),
        "model_revision": model.get("revision"),
        "template_revision": model.get("template_revision"),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip("/")
    report: dict[str, Any] = {
        "schema": "szl.model-inference-lab.live-verification/v2",
        "state": "FAILED",
        "base_url_sha256": text_sha256(base_url),
        "expected_source_revision": args.expected_source_revision,
        "prompt_sha256": text_sha256(PROMPT),
        "prompt_or_output_text_persisted": False,
    }
    try:
        require(
            len(args.expected_source_revision) == 40
            and all(character in "0123456789abcdef" for character in args.expected_source_revision),
            "expected source revision must be a lowercase 40-character Git SHA",
        )
        health, health_headers = wait_for_ready(
            base_url,
            expected_source_revision=args.expected_source_revision,
        )
        verify_health(health, health_headers, args.expected_source_revision)
        status, contract, _ = request_json(
            f"{base_url}/.well-known/szl-governed-inference-contract.json",
            timeout=30.0,
        )
        require(status == 200, "governed contract endpoint failed")
        verify_contract(contract)

        last_status = 0
        inference: dict[str, Any] = {}
        inference_headers: dict[str, str] = {}
        for attempt in range(1, 4):
            last_status, inference, inference_headers = request_json(
                f"{base_url}/api/v2/governed-infer",
                payload={"prompt": PROMPT, "max_new_tokens": 32, "k": 3},
                timeout=120.0,
            )
            if last_status == 200:
                break
            if attempt != 3:
                time.sleep(15.0)
        require(last_status == 200, f"governed inference failed: HTTP {last_status}")
        evidence = verify_inference(
            inference,
            inference_headers,
            args.expected_source_revision,
        )
        status, anatomy, _ = request_json(
            f"{base_url}/api/v2/anatomy/last", timeout=30.0
        )
        require(status == 200, "Anatomy readback endpoint failed")
        require(anatomy.get("observer_authority") == "NONE", "Anatomy readback authority drift")
        require((anatomy.get("observation_count") or 0) >= 1, "Anatomy observed no inference")
        last_event = anatomy.get("last") or {}
        require(last_event.get("output_sha256") == evidence["output_sha256"], "Anatomy/output binding mismatch")
        require(not walk_banned_keys(anatomy), "Anatomy persisted a forbidden field")

        report.update(
            {
                "state": "VERIFIED",
                "controller_revision": FORGE_CONTROLLER_REVISION,
                "second_brain_revision": SECOND_BRAIN_REVISION,
                "nemo_revision": NEMO_REVISION,
                "source_revision": args.expected_source_revision,
                "second_brain_public_chunk_count": 575,
                "locked_proven_formula_count": 8,
                "runtime_winner": "UNSELECTED",
                "tool_execution": False,
                "anatomy_observation_count": anatomy.get("observation_count"),
                "evidence": evidence,
            }
        )
        write_report(args.report, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:500]
        write_report(args.report, report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
