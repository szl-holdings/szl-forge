#!/usr/bin/env python3
"""Verify the deployed governed inference surface without retaining prompt/output text."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# This verifier is also loaded with runpy.run_path() from the repository root.
# Resolve the policy beside this file, independent of the caller's sys.path.
_policy_path = Path(__file__).resolve().with_name("public_transcript_export.py")
_policy_spec = importlib.util.spec_from_file_location(
    "szl_forge_public_transcript_export", _policy_path
)
if _policy_spec is None or _policy_spec.loader is None:
    raise ImportError(f"public export policy is unavailable: {_policy_path}")
_policy = importlib.util.module_from_spec(_policy_spec)
_policy_spec.loader.exec_module(_policy)
PublicExportHold = _policy.PublicExportHold
validate_public_body = _policy.validate_public_body

PROMPT = "Answer with exactly this status, then cite evidence: Lambda remains Conjecture 1, advisory only."
FORGE_CONTROLLER_REVISION = "9f227f6a10dac178b29130c742c98451b6ed8391"
SECOND_BRAIN_REVISION = "1d3960c69235f117b7ec2b5ea97472f81fb588f5"
NEMO_REVISION = "810231a531188bb569e3faa17396386eb0a5e260"
MODEL_REVISION = "67d60ec577730747055491640cfb91fc4a4b5d25"
LOCKED_EIGHT = ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"]
ATLAS_REPOSITORY = "szl-holdings/szl-formulas"
ATLAS_REVISION = "d03536144d2f23ee2da35c356e2558abe5183779"
ATLAS_GIT_BLOB_SHA = "1d3737cb3fc32ee17916f9dfe139a286a930a29f"
ATLAS_SHA256 = "765b9e5c9dd8d4fc5c7518a5b2f70cd6008918591cfa5edfc9868453cb30b7a2"
ATLAS_PAYLOAD_SHA256 = "d7b08cde24f57a40de5707c4336ae41eae48aff9e91cb210ec3152ae8d3a8024"
ATTRIBUTED_SOURCE = {
    "repository": "szl-holdings/szl-formula-ledger",
    "revision": "ceaef540eba6c5acf85091faf4a20cd9aef480f9",
    "path": "formulas/corpus.json",
    "git_blob_sha": "600654f89cb062320acae4284cb5220ff881eb3c",
    "sha256": "c0b6dfee3233097307c518a57c076e5ba3ea9013f34ab454bef92aeddf0f3634",
    "state": "ARCHIVED_ATTRIBUTED_SOURCE",
}
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


def verify_public_export(result: dict[str, Any]) -> str:
    """Check that the live response is the public projection, not a continuation."""
    require("continuation" not in result, "private continuation reached public response")
    receipt = result.get("public_export_receipt")
    require(isinstance(receipt, dict), "public export receipt missing")
    require(
        set(receipt)
        == {"schema", "state", "policy", "source_schema", "public_sha256", "signature_status"},
        "public export receipt shape drift",
    )
    require(
        receipt.get("schema") == "szl.forge.public-transcript-export-receipt/v1"
        and receipt.get("state") == "EXPORTED"
        and receipt.get("policy") == "strict-v1-private-continuation-omitted"
        and receipt.get("source_schema") == "szl.forge.production-governed-inference/v2"
        and receipt.get("signature_status") == "UNSIGNED_LOCAL",
        "public export receipt metadata drift",
    )
    public_without_receipt = dict(result)
    del public_without_receipt["public_export_receipt"]
    try:
        validate_public_body(public_without_receipt)
    except PublicExportHold as exc:
        raise VerificationError("public export schema drift") from exc
    expected_digest = canonical_sha256(public_without_receipt)
    require(receipt.get("public_sha256") == expected_digest, "public export digest mismatch")
    return expected_digest


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
    frontier = brain.get("frontier") or {}
    require(frontier.get("ready") is True, "frontier index is not ready")
    require(frontier.get("source_count") == 7, "frontier source count drift")
    require(frontier.get("state") == "REVIEW_REQUIRED", "frontier review boundary drift")
    require(frontier.get("training_authority") == "NONE", "frontier gained training authority")
    require(frontier.get("execution_authority") == "NONE", "frontier gained execution authority")
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
        ("szl-second-brain", "1.3.0"),
        ("szl-nemo", "0.4.0"),
    ):
        observed = packages.get(name) or {}
        require(observed.get("match") is True, f"{name} package mismatch")
        require(observed.get("observed") == expected, f"{name} version mismatch")
    dependency = health.get("dependency_status") or {}
    require(dependency.get("formula_atlas_ready") is True, "Formula Atlas is not ready")
    atlas = health.get("formula_atlas") or {}
    require(atlas.get("ready") is True, "Formula Atlas health is not ready")
    for name, expected in (
        ("attributed_formula_count", 30),
        ("executable_formula_count", 21),
        ("quant_domain_count", 9),
        ("locked_proven_count", 8),
    ):
        require(atlas.get(name) == expected, f"Formula Atlas {name} drift")
    require(atlas.get("formula_may_authorize") is False, "formula gained authority")
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
    require(
        contract.get("schema") == "szl.model-inference-lab.governed-contract/v3",
        "governed contract schema drift",
    )
    endpoint = contract.get("endpoint") or {}
    require(
        endpoint.get("path") == "/api/v3/governed-infer"
        and endpoint.get("response_schema")
        == "szl.forge.public-governed-inference/v3",
        "governed endpoint path drift",
    )
    require(endpoint.get("tools") is False, "public tools must remain disabled")
    require(
        contract.get("retired_public_post")
        == {
            "path": "/api/v2/governed-infer",
            "status_code": 410,
            "continuation_available": False,
        },
        "v2 public POST retirement drift",
    )
    require(
        contract.get("public_read_only_endpoints")
        == {
            "formula_atlas": "/api/v2/formula-atlas",
            "source": "/api/source",
            "well_known_source": "/.well-known/szl-source.json",
        },
        "public Formula Atlas route contract drift",
    )
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


def verify_formula_atlas(atlas: dict[str, Any]) -> dict[str, Any]:
    require(atlas.get("schema") == "szl.formula-atlas.public/v2", "atlas schema drift")
    require(
        atlas.get("state") == "VERIFIED_IMMUTABLE_PUBLIC_PROJECTION",
        "atlas projection is unavailable",
    )
    sources = atlas.get("source") or {}
    materialized = sources.get("atlas") or {}
    require(materialized.get("repository") == ATLAS_REPOSITORY, "atlas repo drift")
    require(materialized.get("revision") == ATLAS_REVISION, "atlas revision drift")
    require(materialized.get("git_blob_sha") == ATLAS_GIT_BLOB_SHA, "atlas blob drift")
    require(materialized.get("sha256") == ATLAS_SHA256, "atlas digest drift")
    require(
        materialized.get("payload_sha256") == ATLAS_PAYLOAD_SHA256,
        "atlas payload digest drift",
    )
    require(
        sources.get("attributed_corpus") == ATTRIBUTED_SOURCE,
        "attributed corpus source drift",
    )
    counts = atlas.get("counts") or {}
    for name, expected in (
        ("attributed_formula_count", 30),
        ("executable_formula_count", 21),
        ("quant_domain_count", 9),
        ("locked_proven_count", 8),
    ):
        require(counts.get(name) == expected, f"atlas {name} drift")
    handles = atlas.get("formula_handles") or []
    require(len(handles) == 30, "formula handle count drift")
    require(
        len({item.get("id") for item in handles}) == 30,
        "duplicate formula handles",
    )
    require(
        all(
            set(item)
            == {
                "id",
                "class",
                "reported_status",
                "quant_domain",
                "admission",
                "locked_proven_membership",
                "source_attribution",
            }
            for item in handles
        ),
        "formula handle projection exposed unexpected fields",
    )
    require(
        len(atlas.get("executable_formula_handles") or []) == 21,
        "executable formula handle count drift",
    )
    require(len(atlas.get("quant_domains") or []) == 9, "quant domain count drift")
    proof = atlas.get("proof_authority") or {}
    require(
        set(proof.get("locked_proven_ids") or []) == set(LOCKED_EIGHT),
        "locked formula identity drift",
    )
    require(
        proof.get("lambda_status") == "CONJECTURE_1_OPEN_ADVISORY_ONLY",
        "Lambda status drift",
    )
    authority = atlas.get("authority") or {}
    require(authority.get("formula_may_authorize") is False, "formula gained authority")
    require(authority.get("model_may_authorize") is False, "model gained authority")
    require(
        authority.get("public_effectors_enabled") is False,
        "public effectors became enabled",
    )
    require(authority.get("public_tools") == [], "public tools are not empty")
    privacy = atlas.get("privacy") or {}
    for key in (
        "formula_statements_included",
        "second_brain_handles_included",
        "private_graph_content_included",
        "private_retrieval_performed",
    ):
        require(privacy.get(key) is False, f"atlas privacy boundary drift: {key}")
    require(not walk_banned_keys(atlas), "atlas exposed private content")
    return {
        "atlas_revision": ATLAS_REVISION,
        "atlas_sha256": ATLAS_SHA256,
        "atlas_payload_sha256": ATLAS_PAYLOAD_SHA256,
        "attributed_corpus_revision": ATTRIBUTED_SOURCE["revision"],
        "attributed_corpus_sha256": ATTRIBUTED_SOURCE["sha256"],
    }


def verify_source_contract(
    source: dict[str, Any],
    expected_source_revision: str,
    expected_release_manifest_sha256: str,
) -> None:
    require(
        source.get("schema") == "szl.model-inference-lab.source/v1",
        "source contract schema drift",
    )
    require(source.get("state") == "OBSERVED", "source contract is not observed")
    service = source.get("service") or {}
    require(
        service.get("repository") == "szl-holdings/szl-forge",
        "source repository drift",
    )
    require(
        service.get("revision") == expected_source_revision,
        "source revision mismatch",
    )
    require(
        bool(service.get("release_id")), "source release identity missing"
    )
    require(
        service.get("release_manifest_sha256") == expected_release_manifest_sha256,
        "source release manifest digest mismatch",
    )
    components = source.get("components") or {}
    require(
        (components.get("formula_atlas") or {}).get("revision") == ATLAS_REVISION,
        "source Formula Atlas revision drift",
    )
    require(
        components.get("attributed_formula_corpus") == ATTRIBUTED_SOURCE,
        "source attributed corpus drift",
    )
    authority = source.get("authority") or {}
    require(authority.get("formula_may_authorize") is False, "source formula gained authority")
    require(authority.get("model_may_authorize") is False, "source model gained authority")
    require(
        authority.get("public_effectors_enabled") is False,
        "source effectors enabled",
    )
    require(authority.get("public_tools") == [], "source public tools are not empty")
    require(not walk_banned_keys(source), "source contract exposed private content")


def verify_inference(
    result: dict[str, Any],
    headers: dict[str, str],
    expected_source_revision: str,
) -> dict[str, Any]:
    public_export_digest = verify_public_export(result)
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
        headers.get("x-szl-governed-inference") == "v3",
        "governed response header missing",
    )
    return {
        "public_export_sha256": public_export_digest,
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


def verify_retired_public_post(
    status: int, result: dict[str, Any], headers: dict[str, str]
) -> None:
    """The superseded public route must not invoke or serialize the controller."""
    require(status == 410, "v2 public POST was not retired")
    require(
        result
        == {
            "schema": "szl.model-inference-lab.governed-endpoint-retired/v1",
            "state": "BLOCKED",
            "reason_code": "PUBLIC_V2_CONTINUATION_RETIRED",
            "successor": "/api/v3/governed-infer",
        },
        "v2 retirement body drift",
    )
    require(
        headers.get("x-szl-governed-inference") == "v2"
        and headers.get("cache-control") == "no-store",
        "v2 retirement headers drift",
    )


def retired_post_probe_payload() -> dict[str, Any]:
    """The old v2 request model rejects this before its controller is invoked."""
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-release-manifest-sha256", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip("/")
    report: dict[str, Any] = {
        "schema": "szl.model-inference-lab.live-verification/v3",
        "state": "FAILED",
        "base_url_sha256": text_sha256(base_url),
        "expected_source_revision": args.expected_source_revision,
        "expected_release_manifest_sha256": args.expected_release_manifest_sha256,
        "prompt_sha256": text_sha256(PROMPT),
        "prompt_or_output_text_persisted": False,
    }
    try:
        require(
            len(args.expected_source_revision) == 40
            and all(character in "0123456789abcdef" for character in args.expected_source_revision),
            "expected source revision must be a lowercase 40-character Git SHA",
        )
        require(
            len(args.expected_release_manifest_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in args.expected_release_manifest_sha256
            ),
            "expected release manifest digest must be lowercase SHA-256",
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

        status, formula_atlas, _ = request_json(
            f"{base_url}/api/v2/formula-atlas", timeout=30.0
        )
        require(status == 200, "Formula Atlas endpoint failed")
        formula_evidence = verify_formula_atlas(formula_atlas)

        status, source_contract, _ = request_json(
            f"{base_url}/api/source", timeout=30.0
        )
        require(status == 200, "source contract endpoint failed")
        verify_source_contract(
            source_contract,
            args.expected_source_revision,
            args.expected_release_manifest_sha256,
        )
        status, well_known_source, _ = request_json(
            f"{base_url}/.well-known/szl-source.json", timeout=30.0
        )
        require(status == 200, "well-known source endpoint failed")
        require(
            source_contract == well_known_source,
            "source contract aliases returned different payloads",
        )

        retired_status, retired_body, retired_headers = request_json(
            f"{base_url}/api/v2/governed-infer",
            payload=retired_post_probe_payload(),
            timeout=30.0,
        )
        verify_retired_public_post(retired_status, retired_body, retired_headers)

        last_status = 0
        inference: dict[str, Any] = {}
        inference_headers: dict[str, str] = {}
        for attempt in range(1, 4):
            last_status, inference, inference_headers = request_json(
                f"{base_url}/api/v3/governed-infer",
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
                "formula_atlas": formula_evidence,
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
