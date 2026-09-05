import json
import os
import unittest
from pathlib import Path
from unittest import mock

import app
import app_governed as governed
from inference import ProductionBoundaryError


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
        with mock.patch.dict(
            os.environ,
            {app.SOURCE_REVISION_ENV: "f" * 40},
            clear=False,
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
        controlled = {
            "state": "PROPOSAL",
            "schema": "szl.forge.production-governed-inference/v2",
            "executed": False,
        }
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

    def test_release_manifest_covers_governed_runtime_and_live_verifier(self):
        manifest = app.load_release_manifest()
        for path in (
            "app_governed.py",
            "tests/test_governed_app.py",
            "verify_governed_live.py",
        ):
            self.assertIn(path, manifest["source_files"])


if __name__ == "__main__":
    unittest.main()
