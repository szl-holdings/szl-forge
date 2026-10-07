import copy
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import app
import app_governed as governed
import verify_governed_live as live_verifier
from inference import ProductionBoundaryError


def synthetic_controller_result():
    """Run the real controller with synthetic, in-process components only."""
    evidence_text = "Synthetic public evidence about the locked eight."
    evidence_digest = governed.text_sha256(evidence_text)
    evidence = [{"node_id": "synthetic:1", "source": "synthetic", "sha256": evidence_digest}]
    source_revision = "f" * 40

    def retriever(query, k=6):
        return {
            "ready": True,
            "content_access": "HANDLES_ONLY",
            "handles": [
                {"nodeId": "synthetic:1", "nodeKind": "INDEX", "label": "SAMPLE", "note": "synthetic"}
            ],
            "evidence": evidence,
            "evidence_set_sha256": governed.canonical_sha256(evidence),
            "ranking_receipt": {"mode": "SYNTHETIC", "k": k},
        }

    def hydrator(handles, *, principal_id, tenant_id, policy_revision):
        return {
            "state": "AUTHORIZED_CONTENT_READY",
            "ready": True,
            "content_access": "CONTROLLER_ONLY",
            "principal_id_sha256": governed.text_sha256(principal_id),
            "tenant_id_sha256": governed.text_sha256(tenant_id),
            "policy_revision": policy_revision,
            "evidence_set_sha256": governed.canonical_sha256(evidence),
            "documents": [{**evidence[0], "content": evidence_text}],
            "raw_graph_nodes_admitted_to_gradients": 0,
        }

    class Generator:
        def identity(self):
            return {
                "model": {
                    "id": app.OPENAI_MODEL_ID,
                    "revision": app.MODEL_REVISION,
                    "adapter_revision": "NONE",
                    "tokenizer_revision": app.MODEL_REVISION,
                    "template_revision": source_revision,
                    "quantization_revision": "sha256:" + app.MODEL_SHA256,
                },
                "runtime": {
                    "engine": "synthetic",
                    "version": "1.0",
                    "hardware_fingerprint": "synthetic:cpu",
                },
            }

        def __call__(self, context):
            answer = "Synthetic résumé [synthetic:1]."
            return {
                "text": answer,
                "output_schema": "szl.answer-with-node-citations/v2",
                "claims": [
                    {
                        "label": "MODELED",
                        "statement_sha256": governed.text_sha256(answer),
                        "supporting_node_ids": ["synthetic:1"],
                    }
                ],
                "citations": ["synthetic:1"],
                "identity": self.identity(),
                "metrics": {
                    "elapsed_ms": 1.0,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "finish_reason": "stop",
                    "nemo_text_witness": {
                        "decision": "ALLOW",
                        "rule_version": "doctrine-v11/R1-R5",
                        "input_hash": "sha256:" + "a" * 64,
                        "violated_rules": [],
                        "reasons": [],
                    },
                },
            }

    def witness(envelope):
        return {
            "decision": "ALLOW",
            "rule_version": "doctrine-v11/E1-E10",
            "input_hash": "sha256:" + governed.canonical_sha256(envelope),
            "violated_rules": [],
            "reasons": [],
        }

    return governed.production_infer(
        {
            "request_id": "synthetic-public-projection",
            "prompt": "Summarize synthetic evidence.",
            "principal_id": governed.PUBLIC_PRINCIPAL,
            "tenant_id": governed.PUBLIC_TENANT,
            "policy_revision": governed.PUBLIC_POLICY_REVISION,
            "grounding_required": True,
            "formula_applications": [],
        },
        retriever=retriever,
        hydrator=hydrator,
        generator=Generator(),
        witness=witness,
        observer=lambda event: None,
    )


class GovernedSpaceTests(unittest.TestCase):
    def setUp(self):
        self.original_state = dict(app.state)
        self.original_anatomy = dict(governed._anatomy_state)

    def tearDown(self):
        app.state.clear()
        app.state.update(self.original_state)
        governed._anatomy_state.clear()
        governed._anatomy_state.update(self.original_anatomy)

    def test_public_policy_and_dependency_revisions_are_immutable(self):
        self.assertTrue(governed.PUBLIC_POLICY_REVISION.startswith("sha256:"))
        self.assertEqual(len(governed.PUBLIC_POLICY_REVISION), 71)
        self.assertEqual(
            governed.FORGE_CONTROLLER_REVISION,
            "9f227f6a10dac178b29130c742c98451b6ed8391",
        )
        self.assertEqual(
            governed.SECOND_BRAIN_REVISION,
            "1d3960c69235f117b7ec2b5ea97472f81fb588f5",
        )
        self.assertEqual(
            governed.NEMO_REVISION,
            "810231a531188bb569e3faa17396386eb0a5e260",
        )

    def test_public_authorizer_is_fixed_scope_only(self):
        self.assertTrue(
            governed._public_authorizer(
                governed.PUBLIC_PRINCIPAL,
                governed.PUBLIC_TENANT,
                governed.PUBLIC_POLICY_REVISION,
                "node-1",
                "doc",
            )
        )
        for principal, tenant, policy in (
            ("other", governed.PUBLIC_TENANT, governed.PUBLIC_POLICY_REVISION),
            (governed.PUBLIC_PRINCIPAL, "private", governed.PUBLIC_POLICY_REVISION),
            (governed.PUBLIC_PRINCIPAL, governed.PUBLIC_TENANT, "a" * 40),
        ):
            with self.subTest(principal=principal, tenant=tenant, policy=policy):
                self.assertFalse(
                    governed._public_authorizer(
                        principal,
                        tenant,
                        policy,
                        "node-1",
                        "doc",
                    )
                )

    def test_request_is_proposal_only_and_rejects_extra_fields(self):
        value = governed.GovernedInferenceRequest(prompt="  explain Lambda  ")
        self.assertEqual(value.prompt, "explain Lambda")
        self.assertLessEqual(value.k, governed.MAX_GOVERNED_K)
        with self.assertRaises(ValueError):
            governed.GovernedInferenceRequest(
                prompt="hello",
                tool_intent={"tool": "repo.write"},
            )
        for token in app.RESERVED_CHAT_TOKENS:
            with self.assertRaises(ValueError):
                governed.GovernedInferenceRequest(prompt=f"hello {token}")

    def test_space_generator_binds_model_template_quantization_and_hardware(self):
        app.state.update(
            {"status": "READY", "llama_cpp_version": "0.3.21"}
        )
        with (
            mock.patch.dict(
                os.environ,
                {app.SOURCE_REVISION_ENV: "f" * 40},
                clear=False,
            ),
            mock.patch.object(governed.platform, "machine", return_value="synthetic-cpu"),
        ):
            identity = governed.SpaceLlamaGenerator(
                24,
                text_witness=lambda *_args: {"decision": "ALLOW"},
            ).identity()
        model = identity["model"]
        self.assertEqual(model["revision"], app.MODEL_REVISION)
        self.assertEqual(model["tokenizer_revision"], app.MODEL_REVISION)
        self.assertEqual(model["template_revision"], "f" * 40)
        self.assertEqual(
            model["quantization_revision"], "sha256:" + app.MODEL_SHA256
        )
        self.assertEqual(identity["runtime"]["engine"], "llama-cpp-python")
        self.assertTrue(
            identity["runtime"]["hardware_fingerprint"].startswith("sha256:")
        )

    def test_space_generator_uses_existing_bounded_path_and_real_text_gate(self):
        app.state.update(
            {"status": "READY", "llama_cpp_version": "0.3.21"}
        )
        answer = "Lambda remains a conjecture [node-1]."
        text_decision = {
            "decision": "ALLOW",
            "rule_version": "doctrine-v11/R1-R5",
            "input_hash": "sha256:" + ("a" * 64),
            "violated_rules": [],
            "reasons": [],
        }
        context = {
            "prompt": "Explain Lambda.",
            "formula_applications": [],
            "authority": "PROPOSAL_ONLY",
            "evidence": [
                {
                    "node_id": "node-1",
                    "source": "formula",
                    "sha256": "b" * 64,
                    "content": "Lambda uniqueness is Conjecture 1.",
                }
            ],
        }
        with (
            mock.patch.dict(
                os.environ,
                {app.SOURCE_REVISION_ENV: "f" * 40},
                clear=False,
            ),
            mock.patch.object(
                app,
                "run_bounded_completion",
                return_value={
                    "output": answer,
                    "elapsed_ms": 10,
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "finish_reason": "stop",
                },
            ) as bounded,
        ):
            result = governed.SpaceLlamaGenerator(
                24,
                text_witness=lambda *_args: text_decision,
            )(context)
        bounded.assert_called_once()
        self.assertEqual(result["text"], answer)
        self.assertEqual(result["citations"], ["node-1"])
        self.assertEqual(
            result["claims"][0]["supporting_node_ids"], ["node-1"]
        )
        self.assertEqual(
            result["metrics"]["nemo_text_witness"]["decision"], "ALLOW"
        )

    def test_text_witness_block_prevents_output_from_entering_controller(self):
        app.state.update(
            {"status": "READY", "llama_cpp_version": "0.3.21"}
        )
        with (
            mock.patch.dict(
                os.environ,
                {app.SOURCE_REVISION_ENV: "f" * 40},
                clear=False,
            ),
            mock.patch.object(
                app,
                "run_bounded_completion",
                return_value={
                    "output": "Lambda is a proven theorem.",
                    "elapsed_ms": 10,
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "finish_reason": "stop",
                },
            ),
        ):
            with self.assertRaisesRegex(
                ProductionBoundaryError,
                "R1-R5 blocked or reviewed",
            ):
                governed.SpaceLlamaGenerator(
                    24,
                    text_witness=lambda *_args: {
                        "decision": "BLOCK",
                        "rule_version": "doctrine-v11/R1-R5",
                        "input_hash": "sha256:" + ("a" * 64),
                        "violated_rules": ["R4_lambda_not_theorem"],
                        "reasons": ["Lambda remains a conjecture."],
                    },
                )(
                    {
                        "prompt": "Explain Lambda.",
                        "formula_applications": [],
                        "authority": "PROPOSAL_ONLY",
                        "evidence": [
                            {
                                "node_id": "node-1",
                                "source": "formula",
                                "sha256": "b" * 64,
                                "content": "Lambda uniqueness is Conjecture 1.",
                            }
                        ],
                    }
                )

    def test_local_anatomy_observer_stores_only_sanitized_events(self):
        event = {
            "schema": "szl.anatomy.production-inference-observation/v2",
            "request_id": "req-1",
            "prompt_sha256": "a" * 64,
            "output_sha256": "b" * 64,
            "raw_prompt_present": False,
            "hydrated_content_present": False,
            "private_reasoning_present": False,
            "observer_authority": "NONE",
        }
        governed._observe_anatomy(event)
        payload = json.loads(governed.anatomy_last().body)
        self.assertEqual(payload["observation_count"], 1)
        self.assertEqual(payload["last"]["observer_authority"], "NONE")
        self.assertNotIn("prompt", payload["last"])
        with self.assertRaisesRegex(
            ProductionBoundaryError, "unsafe Anatomy observation"
        ):
            governed._observe_anatomy({"prompt": "do not persist"})

    def test_governed_contract_keeps_runtime_unselected_and_tools_disabled(self):
        payload = json.loads(governed.governed_contract().body)
        self.assertEqual(
            payload["endpoint"]["path"], "/api/v2/governed-infer"
        )
        self.assertFalse(payload["endpoint"]["tools"])
        self.assertEqual(
            payload["runtime_selection"]["winner"], "UNSELECTED"
        )
        self.assertEqual(
            payload["formula_authority"]["locked_proven_count"], 8
        )
        self.assertEqual(
            payload["nemo"]["text_witness"], "doctrine-v11/R1-R5"
        )

    def test_governed_health_requires_model_source_dependencies_and_brain(self):
        app.state.update(
            {"status": "READY", "llama_cpp_version": "0.3.21"}
        )
        components = {
            "retriever": lambda *_args: {
                "ready": True,
                "content_access": "HANDLES_ONLY",
                "handles": [{"nodeId": "node-1"}],
                "corpus_n": 575,
            }
        }
        with (
            mock.patch.dict(
                os.environ,
                {app.SOURCE_REVISION_ENV: "f" * 40},
                clear=False,
            ),
            mock.patch.object(
                governed,
                "_dependency_status",
                return_value={"ready": True, "packages": {}, "contract_ready": True},
            ),
            mock.patch.object(governed, "_components", return_value=components),
            mock.patch.object(governed, "frontier_status", return_value={
                "ready": True, "state": "REVIEW_REQUIRED", "source_count": 7,
                "training_authority": "NONE", "execution_authority": "NONE",
            }),
        ):
            response = governed.governed_health()
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "READY")
        self.assertEqual(payload["second_brain"]["public_chunk_count"], 575)
        self.assertEqual(payload["second_brain"]["frontier"]["source_count"], 7)
        self.assertEqual(payload["second_brain"]["frontier"]["state"], "REVIEW_REQUIRED")

    def test_unavailable_frontier_blocks_governed_health(self):
        app.state.update({"status": "READY", "llama_cpp_version": "0.3.21"})
        components = {"retriever": lambda *_args: {
            "ready": True, "content_access": "HANDLES_ONLY",
            "handles": [{"nodeId": "node-1"}], "corpus_n": 575,
        }}
        with (
            mock.patch.dict(os.environ, {app.SOURCE_REVISION_ENV: "f" * 40}),
            mock.patch.object(governed, "_dependency_status", return_value={"ready": True}),
            mock.patch.object(governed, "_components", return_value=components),
            mock.patch.object(governed, "frontier_status", return_value={"ready": False}),
        ):
            response = governed.governed_health()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["status"], "UNAVAILABLE")

    def test_governed_endpoint_passes_only_fixed_public_scope_to_controller(self):
        app.state.update(
            {"status": "READY", "llama_cpp_version": "0.3.21"}
        )
        controlled = synthetic_controller_result()
        self.assertEqual(controlled["state"], "PROPOSAL")
        controlled["continuation"]["reasoning_blob"] = "SYNTHETIC_PRIVATE_CONTINUATION"
        with (
            mock.patch.dict(
                os.environ,
                {app.SOURCE_REVISION_ENV: "f" * 40},
                clear=False,
            ),
            mock.patch.object(
                governed,
                "_components",
                return_value={
                    "retriever": object(),
                    "hydrator": object(),
                    "witness": object(),
                },
            ),
            mock.patch.object(
                governed,
                "production_infer",
                return_value=controlled,
            ) as infer,
        ):
            response = governed.governed_infer(
                governed.GovernedInferenceRequest(
                    prompt="Explain the current Lambda status.",
                    max_new_tokens=16,
                    k=2,
                )
            )
        self.assertEqual(response.status_code, 200)
        public = json.loads(response.body)
        self.assertNotIn("continuation", public)
        self.assertNotIn("SYNTHETIC_PRIVATE_CONTINUATION", response.body.decode())
        self.assertEqual(public["public_export_receipt"]["state"], "EXPORTED")
        self.assertEqual(
            controlled["continuation"]["reasoning_blob"],
            "SYNTHETIC_PRIVATE_CONTINUATION",
        )
        payload = infer.call_args.args[0]
        self.assertEqual(payload["principal_id"], governed.PUBLIC_PRINCIPAL)
        self.assertEqual(payload["tenant_id"], governed.PUBLIC_TENANT)
        self.assertEqual(
            payload["policy_revision"], governed.PUBLIC_POLICY_REVISION
        )
        self.assertEqual(payload["formula_applications"], [])
        self.assertNotIn("tool_intent", payload)
        self.assertIsInstance(
            infer.call_args.kwargs["generator"],
            governed.SpaceLlamaGenerator,
        )

    def test_public_export_holds_unknown_nested_provider_fields_without_echo(self):
        unsafe = synthetic_controller_result()
        self.assertEqual(unsafe["state"], "PROPOSAL")
        unsafe["metrics"]["encrypted_reasoning"] = "SYNTHETIC_OPAQUE_BLOB"
        unsafe["continuation"]["private"] = "SYNTHETIC_PRIVATE_CONTINUATION"
        with self.assertRaises(governed.PublicExportHold):
            governed.project_public_result(unsafe)
        public = governed.hold_public_result()
        serialized = json.dumps(public)
        self.assertEqual(public["state"], "HOLD")
        self.assertFalse(public["publication_eligible"])
        self.assertNotIn("SYNTHETIC_OPAQUE_BLOB", serialized)
        self.assertNotIn("SYNTHETIC_PRIVATE_CONTINUATION", serialized)
        self.assertEqual(
            unsafe["metrics"]["encrypted_reasoning"], "SYNTHETIC_OPAQUE_BLOB"
        )

    def test_governed_endpoint_holds_opaque_provider_response(self):
        app.state.update({"status": "READY", "llama_cpp_version": "0.3.21"})
        unsafe = synthetic_controller_result()
        unsafe["metrics"]["thinking_signature"] = "SYNTHETIC_OPAQUE_BLOB"
        with (
            mock.patch.dict(os.environ, {app.SOURCE_REVISION_ENV: "f" * 40}),
            mock.patch.object(
                governed,
                "_components",
                return_value={"retriever": object(), "hydrator": object(), "witness": object()},
            ),
            mock.patch.object(governed, "production_infer", return_value=unsafe),
        ):
            response = governed.governed_infer(
                governed.GovernedInferenceRequest(prompt="Synthetic test prompt")
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["state"], "HOLD")
        self.assertNotIn("SYNTHETIC_OPAQUE_BLOB", response.body.decode())

    def test_governed_endpoint_holds_missing_observation_event(self):
        app.state.update({"status": "READY", "llama_cpp_version": "0.3.21"})
        unsafe = synthetic_controller_result()
        del unsafe["anatomy_observation"]["event"]
        unsafe["continuation"]["reasoning_blob"] = "SYNTHETIC_PRIVATE_CONTINUATION"
        with (
            mock.patch.dict(os.environ, {app.SOURCE_REVISION_ENV: "f" * 40}),
            mock.patch.object(
                governed,
                "_components",
                return_value={"retriever": object(), "hydrator": object(), "witness": object()},
            ),
            mock.patch.object(governed, "production_infer", return_value=unsafe),
        ):
            response = governed.governed_infer(
                governed.GovernedInferenceRequest(prompt="Synthetic test prompt")
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["state"], "HOLD")
        self.assertNotIn("SYNTHETIC_PRIVATE_CONTINUATION", response.body.decode())

    def test_public_export_holds_other_missing_nested_required_fields(self):
        base = synthetic_controller_result()
        cases = []
        missing_claim = copy.deepcopy(base)
        del missing_claim["claims"][0]["supporting_node_ids"]
        cases.append(missing_claim)
        missing_handle = copy.deepcopy(base)
        del missing_handle["evidence_handles"][0]["note"]
        cases.append(missing_handle)
        missing_receipt_formula = copy.deepcopy(base)
        del missing_receipt_formula["receipt"]["payload"]["formula_binding"]["locked_proven_ids"]
        missing_receipt_formula["receipt"]["receipt_sha256"] = governed.canonical_sha256(
            missing_receipt_formula["receipt"]["payload"]
        )
        cases.append(missing_receipt_formula)
        for unsafe in cases:
            with self.subTest(keys=set(unsafe["claims"][0])):
                with self.assertRaises(governed.PublicExportHold):
                    governed.project_public_result(unsafe)

    def test_public_export_holds_unknown_top_level_and_bad_receipt(self):
        base = synthetic_controller_result()
        with self.assertRaises(governed.PublicExportHold):
            governed.project_public_result({**base, "provider_trace": "SYNTHETIC"})
        bad = copy.deepcopy(base)
        bad["receipt"]["receipt_sha256"] = "0" * 64
        with self.assertRaises(governed.PublicExportHold):
            governed.project_public_result(bad)

    def test_public_export_unicode_digest_and_live_verifier(self):
        private = synthetic_controller_result()
        self.assertEqual(private["state"], "PROPOSAL")
        public = governed.project_public_result(private)
        self.assertIn("résumé", public["output"])
        self.assertNotIn("continuation", public)
        self.assertIn("continuation", private)
        body = dict(public)
        export_receipt = body.pop("public_export_receipt")
        utf8_digest = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        escaped_digest = hashlib.sha256(
            json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(export_receipt["public_sha256"], utf8_digest)
        self.assertNotEqual(export_receipt["public_sha256"], escaped_digest)
        self.assertEqual(live_verifier.verify_public_export(public), utf8_digest)
        self.assertEqual(
            live_verifier.verify_inference(
                public, {"x-szl-governed-inference": "v2"}, "f" * 40
            )["public_export_sha256"],
            utf8_digest,
        )

    def test_live_verifier_rejects_public_continuation_or_export_receipt_drift(self):
        public = governed.project_public_result(synthetic_controller_result())
        cases = []
        with_continuation = copy.deepcopy(public)
        with_continuation["continuation"] = {"reasoning_blob": "SYNTHETIC_PRIVATE"}
        cases.append(with_continuation)
        for field, value in (
            ("schema", "wrong"),
            ("policy", "wrong"),
            ("public_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(public)
            changed["public_export_receipt"][field] = value
            cases.append(changed)
        missing = copy.deepcopy(public)
        del missing["public_export_receipt"]
        cases.append(missing)
        opaque_with_recomputed_digest = copy.deepcopy(public)
        opaque_with_recomputed_digest["metrics"]["encrypted_reasoning"] = (
            "SYNTHETIC_OPAQUE_BLOB"
        )
        opaque_body = dict(opaque_with_recomputed_digest)
        del opaque_body["public_export_receipt"]
        opaque_with_recomputed_digest["public_export_receipt"]["public_sha256"] = (
            live_verifier.canonical_sha256(opaque_body)
        )
        cases.append(opaque_with_recomputed_digest)
        for changed in cases:
            with self.subTest(changed=changed.get("public_export_receipt")):
                with self.assertRaises(live_verifier.VerificationError):
                    live_verifier.verify_public_export(changed)
                with self.assertRaises(live_verifier.VerificationError):
                    live_verifier.verify_inference(
                        changed, {"x-szl-governed-inference": "v2"}, "f" * 40
                    )

    def test_public_export_rejects_opaque_known_field_and_receipt_metadata_drift(self):
        base = synthetic_controller_result()
        changed = copy.deepcopy(base)
        changed["metrics"]["finish_reason"] = "SYNTHETIC_OPAQUE_BLOB"
        with self.assertRaises(governed.PublicExportHold):
            governed.project_public_result(changed)
        for path, value in (
            (("algorithm",), "sha3-256"),
            (("canonicalization",), "ascii-json"),
            (("signature", "must_be_signed_before_consequential_action"), False),
        ):
            changed = copy.deepcopy(base)
            target = changed["receipt"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path):
                with self.assertRaises(governed.PublicExportHold):
                    governed.project_public_result(changed)
        changed = copy.deepcopy(base)
        del changed["output"]
        with self.assertRaises(governed.PublicExportHold):
            governed.project_public_result(changed)

    def test_docker_and_requirements_pin_the_governed_entrypoint_and_sources(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn(
            'CMD ["python", "-m", "uvicorn", "app_governed:app"',
            dockerfile,
        )
        for revision in (
            governed.FORGE_CONTROLLER_REVISION,
            governed.SECOND_BRAIN_REVISION,
            governed.NEMO_REVISION,
        ):
            self.assertIn(revision, requirements)
        self.assertNotIn("git+https://", requirements)

    def test_governed_entrypoint_serves_the_landing_design_system(self):
        from fastapi.testclient import TestClient

        client = TestClient(governed.app)
        page = client.get("/")
        stylesheet = client.get("/szl/szl-design-system.css")
        self.assertIn('href="/szl/szl-design-system.css"', page.text)
        self.assertEqual(200, stylesheet.status_code)
        self.assertEqual("text/css; charset=utf-8", stylesheet.headers["content-type"])
        self.assertEqual("nosniff", stylesheet.headers["x-content-type-options"])
        self.assertEqual(
            "image/svg+xml",
            client.get("/szl/logos/szl_favicon.svg").headers["content-type"],
        )
        self.assertEqual(404, client.get("/szl/SOURCE.json").status_code)
        self.assertEqual(404, client.get("/kanchay/kanchay.css").status_code)

    def test_release_manifest_covers_governed_runtime_and_live_verifier(self):
        manifest = app.load_release_manifest()
        for path in (
            "app_governed.py",
            "public_transcript_export.py",
            "tests/test_governed_app.py",
            "verify_governed_live.py",
        ):
            self.assertIn(path, manifest["source_files"])


if __name__ == "__main__":
    unittest.main()
