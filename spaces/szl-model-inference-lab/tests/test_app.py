import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app
import download_artifacts
import verify_execution_record
from fastapi.testclient import TestClient


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class FakeLlm:
    def __init__(self, output="bounded answer", finish_reason="stop"):
        self.cache = None
        self.output = output
        self.finish_reason = finish_reason
        self.tokenize_calls = []
        self.completion_kwargs = None
        self.stream = None

    def tokenize(self, value, *, add_bos, special):
        self.tokenize_calls.append(
            {"text": value.decode("utf-8"), "add_bos": add_bos, "special": special}
        )
        return [1, 2, 3, 4, 5] if special else [6, 7]

    def create_completion(self, **kwargs):
        self.completion_kwargs = kwargs
        self.stream = FakeStream(
            [
                {
                    "choices": [
                        {"text": self.output, "finish_reason": self.finish_reason}
                    ]
                }
            ]
        )
        return self.stream


class ExplodingLlm(FakeLlm):
    def create_completion(self, **kwargs):
        raise RuntimeError("DO_NOT_LEAK_RUNTIME_DETAIL")


class AppContractTests(unittest.TestCase):
    def test_immutable_model_contract(self):
        self.assertEqual(len(app.MODEL_REVISION), 40)
        self.assertEqual(len(app.MODEL_SHA256), 64)
        self.assertEqual(app.MODEL_SIZE, 986_047_904)
        self.assertEqual(app.MAX_NEW_TOKENS, 32)

    def test_prompt_cache_is_structurally_disabled(self):
        safe = FakeLlm()
        app.enforce_prompt_cache_disabled(safe)

        class MissingCacheContract:
            pass

        class EnabledCache:
            cache = object()

        for unsafe in (MissingCacheContract(), EnabledCache()):
            with self.assertRaisesRegex(
                RuntimeError, "PROMPT_CACHE_MUST_REMAIN_DISABLED"
            ):
                app.enforce_prompt_cache_disabled(unsafe)

        payload = app.identity_payload()
        self.assertEqual(
            payload["runtime"]["prompt_cache"]["advisory"], app.DISKCACHE_ADVISORY
        )
        self.assertIn(
            payload["runtime"]["prompt_cache"]["status"],
            {"NOT_CHECKED", "DISABLED"},
        )

    def test_runtime_artifact_path_is_fixed_regular_and_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / app.MODEL_FILE
            model.write_bytes(b"model")
            with mock.patch.object(app, "ARTIFACT_ROOT", root):
                self.assertEqual(model, app.artifact_path(app.MODEL_FILE))
                with self.assertRaisesRegex(RuntimeError, "ARTIFACT_NOT_ALLOWLISTED"):
                    app.artifact_path("other.gguf")
                model.unlink()
                with self.assertRaisesRegex(RuntimeError, "ARTIFACT_NOT_REGULAR"):
                    app.artifact_path(app.MODEL_FILE)

    def test_image_fetches_only_pinned_artifacts_before_runtime_goes_offline(self):
        dockerfile = (app.SOURCE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("MODEL_DIR_OVERRIDE", dockerfile)
        fetch = dockerfile.index(
            "RUN python download_artifacts.py --fetch "
            "--output-dir /home/user/model-artifacts"
        )
        offline = dockerfile.index("HF_HUB_OFFLINE=1")
        self.assertLess(fetch, offline)
        self.assertIn("HF_HUB_DISABLE_XET=1", dockerfile)
        self.assertIn(
            "COPY --from=artifact-builder --chown=root:root "
            "/home/user/model-artifacts/ /opt/szl/model-artifacts/",
            dockerfile,
        )

    def test_build_fetch_is_exact_revision_tokenless_and_network_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc")
            digest = app.sha256_file(path)
            with (
                mock.patch.object(
                    download_artifacts,
                    "ARTIFACTS",
                    {"sample.bin": (3, digest)},
                ),
                mock.patch.object(
                    download_artifacts,
                    "hf_hub_download",
                    return_value=str(path),
                ) as download,
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("MODEL_DIR_OVERRIDE", None)
                output = Path(directory) / "output"
                verified = download_artifacts.verify_artifacts(
                    fetch=True, output_dir=output
                )
        self.assertEqual(verified, (output / "sample.bin",))
        download.assert_called_once_with(
            repo_id=download_artifacts.REPO,
            filename="sample.bin",
            revision=download_artifacts.REVISION,
            local_files_only=False,
            token=False,
        )

    def test_runtime_artifact_verification_stays_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc")
            digest = app.sha256_file(path)
            with (
                mock.patch.object(
                    download_artifacts,
                    "ARTIFACTS",
                    {"sample.bin": (3, digest)},
                ),
                mock.patch.object(
                    download_artifacts,
                    "hf_hub_download",
                    return_value=str(path),
                ) as download,
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("MODEL_DIR_OVERRIDE", None)
                download_artifacts.verify_artifacts()
        download.assert_called_once_with(
            repo_id=download_artifacts.REPO,
            filename="sample.bin",
            revision=download_artifacts.REVISION,
            local_files_only=True,
            token=False,
        )

    def test_build_fetch_rejects_local_override(self):
        with mock.patch.dict(os.environ, {"MODEL_DIR_OVERRIDE": "/models"}):
            with self.assertRaisesRegex(
                RuntimeError, "MODEL_DIR_OVERRIDE is not permitted"
            ):
                download_artifacts.artifact_path(app.MODEL_FILE, fetch=True)

    def test_downloader_and_runtime_artifact_locks_match(self):
        self.assertEqual(download_artifacts.REPO, app.MODEL_REPO)
        self.assertEqual(download_artifacts.REVISION, app.MODEL_REVISION)
        self.assertEqual(
            set(download_artifacts.ARTIFACTS),
            {app.MODEL_FILE, *app.RECEIPT_FILES},
        )
        self.assertEqual(
            download_artifacts.ARTIFACTS[app.MODEL_FILE],
            (app.MODEL_SIZE, app.MODEL_SHA256),
        )
        for filename in app.RECEIPT_FILES:
            self.assertEqual(
                download_artifacts.ARTIFACTS[filename][1],
                app.RECEIPT_SHA256[filename],
            )

    def test_failed_output_verification_does_not_publish_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"abc")
            output = Path(directory) / "output"
            with (
                mock.patch.object(
                    download_artifacts,
                    "ARTIFACTS",
                    {"sample.bin": (3, "0" * 64)},
                ),
                mock.patch.object(
                    download_artifacts,
                    "hf_hub_download",
                    return_value=str(source),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                    download_artifacts.verify_artifacts(
                        fetch=True, output_dir=output
                    )
            self.assertFalse((output / "sample.bin").exists())

    def test_prompt_contract(self):
        self.assertEqual(app.InferenceRequest(prompt="  hello  ").prompt, "hello")
        with self.assertRaises(ValueError):
            app.InferenceRequest(prompt=" ")
        with self.assertRaises(ValueError):
            app.InferenceRequest(prompt="x" * 1_201)
        for token in app.RESERVED_CHAT_TOKENS:
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    app.InferenceRequest(prompt=f"hello {token}")

    def test_generation_budget_is_bounded(self):
        self.assertLessEqual(app.INFERENCE_BUDGET_SECONDS, 45.0)
        self.assertLessEqual(app.MAX_NEW_TOKENS, 32)

    def test_prompt_token_budget_rejects_overflow(self):
        app.enforce_prompt_budget(app.MAX_PROMPT_TOKENS)
        with self.assertRaises(app.HTTPException) as caught:
            app.enforce_prompt_budget(app.MAX_PROMPT_TOKENS + 1)
        self.assertEqual(caught.exception.status_code, 422)

    def test_liveness_separates_starting_from_failed(self):
        original = app.state["status"]
        try:
            app.state["status"] = "STARTING"
            self.assertEqual(app.live().status_code, 200)
            app.state["status"] = "FAILED"
            self.assertEqual(app.live().status_code, 503)
        finally:
            app.state["status"] = original

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abc")
            self.assertEqual(
                app.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_identity_is_explicit_about_receipt_boundary(self):
        payload = app.identity_payload()
        boundary = json.dumps(payload["receipt_boundary"])
        self.assertIn("independent benchmarking", boundary)
        self.assertIn("safety certification", boundary)
        self.assertIn("repository-declared key", boundary)
        self.assertIn("key-ownership", boundary)
        self.assertEqual(len(payload["space"]["release_manifest_sha256"]), 64)

    def test_build_info_fails_closed_without_exact_source_revision(self):
        original = os.environ.pop(app.SOURCE_REVISION_ENV, None)
        try:
            response = app.build_info()
            payload = json.loads(response.body)
            self.assertEqual("UNKNOWN", payload["build"]["state"])
            self.assertIsNone(payload["build"]["revision"])
            self.assertFalse(payload["receipt_minted"])

            os.environ[app.SOURCE_REVISION_ENV] = "not-a-git-sha"
            payload = app.build_info_payload()
            self.assertEqual("UNKNOWN", payload["build"]["state"])
            self.assertIsNone(payload["build"]["revision"])
        finally:
            if original is not None:
                os.environ[app.SOURCE_REVISION_ENV] = original
            else:
                os.environ.pop(app.SOURCE_REVISION_ENV, None)

    def test_build_info_reports_governed_exact_source_revision(self):
        original = os.environ.get(app.SOURCE_REVISION_ENV)
        revision = "a" * 40
        try:
            os.environ[app.SOURCE_REVISION_ENV] = revision
            response = app.build_info()
            payload = json.loads(response.body)
            self.assertEqual("OBSERVED", payload["build"]["state"])
            self.assertEqual(revision, payload["build"]["revision"])
            self.assertEqual("szl.build-info/v1", payload["schema"])
            self.assertFalse(payload["receipt_minted"])
        finally:
            if original is None:
                os.environ.pop(app.SOURCE_REVISION_ENV, None)
            else:
                os.environ[app.SOURCE_REVISION_ENV] = original

    def test_operational_aliases_preserve_liveness_and_readiness(self):
        original = app.state["status"]
        try:
            app.state["status"] = "STARTING"
            self.assertEqual(200, app.healthz().status_code)
            self.assertEqual(503, app.readyz().status_code)
            app.state["status"] = "READY"
            self.assertEqual(200, app.healthz().status_code)
            self.assertEqual(200, app.readyz().status_code)
            app.state["status"] = "FAILED"
            self.assertEqual(503, app.healthz().status_code)
            self.assertEqual(503, app.readyz().status_code)
        finally:
            app.state["status"] = original

    def test_version_fails_closed_without_exact_deploy_revision(self):
        original = os.environ.pop(app.SOURCE_REVISION_ENV, None)
        try:
            response = app.version()
            payload = json.loads(response.body)
            self.assertEqual(503, response.status_code)
            self.assertIsNone(payload["gitSha"])
            self.assertEqual("UNAVAILABLE", payload["evidenceState"])

            os.environ[app.SOURCE_REVISION_ENV] = "b" * 40
            response = app.version()
            payload = json.loads(response.body)
            self.assertEqual(200, response.status_code)
            self.assertEqual("b" * 40, payload["gitSha"])
            self.assertEqual("model-inference", payload["surface"])
        finally:
            if original is None:
                os.environ.pop(app.SOURCE_REVISION_ENV, None)
            else:
                os.environ[app.SOURCE_REVISION_ENV] = original

    def test_evidence_requires_source_and_verified_receipts(self):
        original_revision = os.environ.get(app.SOURCE_REVISION_ENV)
        original_state = dict(app.state)
        try:
            os.environ[app.SOURCE_REVISION_ENV] = "c" * 40
            app.state.update(
                {
                    "status": "READY",
                    "source_integrity": True,
                    "model_sha256": app.MODEL_SHA256,
                    "receipt_status": "DECLARED_KEY_SIGNATURES_VALID",
                    "receipt_evidence": {
                        "training_canonical_sha256": "d" * 64,
                        "eval_canonical_sha256": "e" * 64,
                    },
                }
            )
            response = app.evidence()
            payload = json.loads(response.body)
            self.assertEqual(200, response.status_code)
            self.assertEqual("MEASURED", payload["evidenceState"])
            self.assertEqual(2, len(payload["receipts"]))
            self.assertEqual("UNSIGNED", payload["outputProvenance"]["signatureStatus"])

            app.state["receipt_status"] = "NOT_CHECKED"
            response = app.evidence()
            self.assertEqual(503, response.status_code)
        finally:
            app.state.clear()
            app.state.update(original_state)
            if original_revision is None:
                os.environ.pop(app.SOURCE_REVISION_ENV, None)
            else:
                os.environ[app.SOURCE_REVISION_ENV] = original_revision

    def test_openai_model_catalog_is_immutable_and_unsigned(self):
        response = app.openai_models()
        payload = json.loads(response.body)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], app.OPENAI_MODEL_ID)
        provenance = payload["data"][0]["szl_provenance"]
        self.assertEqual(provenance["output"]["signature_status"], "UNSIGNED")
        self.assertEqual(
            provenance["runtime"]["native_hugging_face_provider_mapping"],
            "NOT_CLAIMED",
        )
        self.assertEqual(response.headers["x-szl-output-signature"], "none")
        self.assertEqual(
            response.headers["x-szl-service-level"], "best-effort-no-sla"
        )

    def test_well_known_contract_advertises_only_bounded_subset(self):
        response = app.inference_contract()
        payload = json.loads(response.body)
        subset = payload["openai_compatible_subset"]
        self.assertFalse(subset["chat_completions"]["streaming"])
        self.assertFalse(subset["chat_completions"]["tools"])
        self.assertEqual(subset["chat_completions"]["choices"], 1)
        self.assertFalse(payload["authentication"]["required_by_application"])
        self.assertEqual(
            payload["authentication"]["client_compatibility_dummy_key"],
            "not-a-secret",
        )
        self.assertIn("Do not send real", payload["authentication"]["warning"])
        self.assertTrue(payload["provenance"]["authenticity_not_established"])
        self.assertEqual(payload["model"]["repo"], app.MODEL_REPO)
        self.assertEqual(payload["model"]["revision"], app.MODEL_REVISION)
        self.assertEqual(payload["model"]["file"], app.MODEL_FILE)
        self.assertEqual(payload["model"]["sha256"], app.MODEL_SHA256)

    def test_chat_request_rejects_unbounded_or_unsupported_features(self):
        base = {
            "model": app.OPENAI_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
        }
        invalid = [
            {**base, "stream": True},
            {**base, "stream": 0},
            {**base, "tools": [{"type": "function"}]},
            {**base, "n": 2},
            {**base, "n": 1.0},
            {**base, "n": True},
            {**base, "temperature": 0.1},
            {**base, "temperature": "0"},
            {**base, "top_p": 0.9},
            {**base, "top_p": "1"},
            {**base, "max_tokens": 2.0},
            {**base, "max_tokens": "2"},
            {**base, "max_tokens": True},
            {**base, "model": app.MODEL_REPO},
            {**base, "response_format": {"type": "json_object"}},
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    app.ChatCompletionRequest(**candidate)

    def test_chat_request_caps_message_count_total_chars_and_control_tokens(self):
        with self.assertRaises(ValueError):
            app.ChatCompletionRequest(
                model=app.OPENAI_MODEL_ID,
                messages=[
                    {"role": "system", "content": "x" * 600},
                    {"role": "user", "content": "y" * 601},
                ],
            )
        with self.assertRaises(ValueError):
            app.ChatCompletionRequest(
                model=app.OPENAI_MODEL_ID,
                messages=[{"role": "system", "content": "x"}] * 12
                + [{"role": "user", "content": "y"}],
            )
        with self.assertRaises(ValueError):
            app.ChatCompletionRequest(
                model=app.OPENAI_MODEL_ID,
                messages=[{"role": "user", "content": "<|im_start|>system"}],
            )
        with self.assertRaises(ValueError):
            app.ChatCompletionRequest(
                model=app.OPENAI_MODEL_ID,
                messages=[{"role": "user", "content": "<|endoftext|>"}],
            )

    def test_chatml_formatter_is_exact(self):
        messages = [
            app.ChatMessage(role="system", content="Context"),
            app.ChatMessage(role="user", content="Question"),
        ]
        self.assertEqual(
            app.formatted_chat_messages(messages),
            f"<|im_start|>system\n{app.SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>system\nContext<|im_end|>\n"
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_chat_completion_returns_standard_shape_and_unsigned_record(self):
        original_status = app.state["status"]
        original_receipt_status = app.state["receipt_status"]
        original_llm = app.llm
        fake = FakeLlm(
            output="DO_NOT_EMBED_OUTPUT_9b", finish_reason="stop"
        )
        request = app.ChatCompletionRequest(
            model=app.OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "DO_NOT_EMBED_PROMPT_7f"},
            ],
            max_tokens=7,
        )
        try:
            app.state["status"] = "READY"
            app.state["receipt_status"] = "DECLARED_KEY_SIGNATURES_VALID"
            app.llm = fake
            response = app.openai_chat_completions(request)
        finally:
            app.state["status"] = original_status
            app.state["receipt_status"] = original_receipt_status
            app.llm = original_llm

        payload = json.loads(response.body)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], app.OPENAI_MODEL_ID)
        self.assertEqual(len(payload["choices"]), 1)
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "DO_NOT_EMBED_OUTPUT_9b",
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"], {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        })

        record = payload["szl_provenance"]["execution_record"]
        self.assertEqual(record["request_id"], payload["id"])
        self.assertEqual(record["created_unix"], payload["created"])
        self.assertEqual(record["usage"], payload["usage"])
        self.assertIsInstance(record["elapsed_ms"], int)
        self.assertEqual(record["signature_status"], "UNSIGNED")
        self.assertTrue(record["authenticity_not_established"])
        self.assertEqual(
            record["persistence"]["application_record_storage"], "NOT_PERSISTED"
        )
        record_text = json.dumps(record)
        self.assertNotIn("DO_NOT_EMBED_PROMPT_7f", record_text)
        self.assertNotIn("DO_NOT_EMBED_OUTPUT_9b", record_text)
        verification = verify_execution_record.verify_payload(
            payload, request.model_dump(mode="json")
        )
        self.assertTrue(verification["hash_matches"])
        self.assertTrue(verification["output_hash_matches"])
        self.assertTrue(verification["request_hash_matches"])
        self.assertTrue(verification["semantic_checks_pass"])
        self.assertEqual(
            response.headers["x-szl-execution-record-sha256"],
            record["record_sha256"],
        )
        self.assertEqual(response.headers["x-szl-output-signature"], "none")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(
            fake.completion_kwargs["prompt"],
            app.formatted_chat_messages(request.messages),
        )
        self.assertEqual(fake.completion_kwargs["max_tokens"], 7)
        self.assertEqual(fake.completion_kwargs["temperature"], 0.0)
        self.assertTrue(fake.completion_kwargs["stream"])
        self.assertTrue(fake.stream.closed)

        tampered_payload = json.loads(response.body)
        tampered_payload["choices"][0]["message"]["content"] += " tampered"
        tampered = verify_execution_record.verify_payload(
            tampered_payload, request.model_dump(mode="json")
        )
        self.assertTrue(tampered["hash_matches"])
        self.assertFalse(tampered["output_hash_matches"])

        changed_scope = dict(record)
        changed_scope["record_sha256_scope"] = "unprotected"
        self.assertFalse(
            verify_execution_record.verify_record(changed_scope)["hash_matches"]
        )

    def test_verifier_normalizes_valid_numeric_forms_and_rejects_extra_fields(self):
        original_status = app.state["status"]
        original_receipt_status = app.state["receipt_status"]
        original_llm = app.llm
        request = app.ChatCompletionRequest(
            model=app.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            top_p=1,
            n=1,
        )
        try:
            app.state["status"] = "READY"
            app.state["receipt_status"] = "DECLARED_KEY_SIGNATURES_VALID"
            app.llm = FakeLlm()
            response = app.openai_chat_completions(request)
        finally:
            app.state["status"] = original_status
            app.state["receipt_status"] = original_receipt_status
            app.llm = original_llm
        payload = json.loads(response.body)
        raw_request = {
            "model": app.OPENAI_MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "top_p": 1,
            "n": 1,
        }
        verified = verify_execution_record.verify_payload(payload, raw_request)
        self.assertTrue(verified["request_hash_matches"])
        negative_zero_request = {**raw_request, "temperature": -0.0}
        verified = verify_execution_record.verify_payload(
            payload, negative_zero_request
        )
        self.assertTrue(verified["request_hash_matches"])
        for field, value in (
            ("response_format", {"type": "json_object"}),
            ("frequency_penalty", 0),
            ("tool_choice", "required"),
            ("parallel_tool_calls", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    verify_execution_record.normalized_request_sha256(
                        {**raw_request, field: value}
                    )

    def test_verifier_binds_outer_response_fields_to_record(self):
        original_status = app.state["status"]
        original_receipt_status = app.state["receipt_status"]
        original_llm = app.llm
        request = app.ChatCompletionRequest(
            model=app.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
        )
        try:
            app.state["status"] = "READY"
            app.state["receipt_status"] = "DECLARED_KEY_SIGNATURES_VALID"
            app.llm = FakeLlm()
            response = app.openai_chat_completions(request)
        finally:
            app.state["status"] = original_status
            app.state["receipt_status"] = original_receipt_status
            app.llm = original_llm
        payload = json.loads(response.body)
        self.assertTrue(
            verify_execution_record.verify_payload(payload)[
                "response_consistency_matches"
            ]
        )
        for field, value in (
            ("id", "chatcmpl-szl-tampered"),
            ("created", 1),
            ("model", "tampered/model"),
            (
                "usage",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            ),
        ):
            with self.subTest(field=field):
                changed = json.loads(response.body)
                changed[field] = value
                verified = verify_execution_record.verify_payload(changed)
                self.assertFalse(verified["response_consistency_matches"])

        adversarial = json.loads(response.body)
        adversarial["szl_provenance"]["output"] = {
            "signature_status": "SIGNED",
            "signature": "forged",
            "termination_reason": "stop",
        }
        self.assertFalse(
            verify_execution_record.verify_payload(adversarial)[
                "response_consistency_matches"
            ]
        )
        adversarial = json.loads(response.body)
        adversarial["szl_provenance"]["model"]["repo"] = "ATTACKER/OTHER"
        self.assertFalse(
            verify_execution_record.verify_payload(adversarial)[
                "response_consistency_matches"
            ]
        )
        adversarial = json.loads(response.body)
        adversarial["choices"][0]["message"]["tool_calls"] = []
        self.assertFalse(
            verify_execution_record.verify_payload(adversarial)[
                "response_consistency_matches"
            ]
        )

    def test_verifier_rejects_self_rehashed_semantic_overclaim(self):
        original_status = app.state["status"]
        original_receipt_status = app.state["receipt_status"]
        original_llm = app.llm
        request = app.ChatCompletionRequest(
            model=app.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
        )
        try:
            app.state["status"] = "READY"
            app.state["receipt_status"] = "DECLARED_KEY_SIGNATURES_VALID"
            app.llm = FakeLlm()
            response = app.openai_chat_completions(request)
        finally:
            app.state["status"] = original_status
            app.state["receipt_status"] = original_receipt_status
            app.llm = original_llm
        record = json.loads(response.body)["szl_provenance"]["execution_record"]
        record["signature_status"] = "SIGNED"
        record["record_sha256"] = verify_execution_record.recompute_record_sha256(
            record
        )
        verified = verify_execution_record.verify_record(record)
        self.assertTrue(verified["hash_matches"])
        self.assertFalse(verified["semantic_checks_pass"])
        self.assertIn("record must remain explicitly unsigned", verified["semantic_errors"])

        record = json.loads(response.body)["szl_provenance"]["execution_record"]
        record["usage"] = {
            "prompt_tokens": 999,
            "completion_tokens": 999,
            "total_tokens": 1998,
        }
        record["record_sha256"] = verify_execution_record.recompute_record_sha256(
            record
        )
        verified = verify_execution_record.verify_record(record)
        self.assertTrue(verified["hash_matches"])
        self.assertFalse(verified["semantic_checks_pass"])
        self.assertIn(
            "record usage exceeds the release token limits",
            verified["semantic_errors"],
        )

        mutations = (
            lambda value: value.__setitem__("attestation", "SIGNED_BY_HUGGING_FACE"),
            lambda value: value["source"].__setitem__(
                "native_provider_mapping", "VERIFIED"
            ),
            lambda value: value["termination"].__setitem__(
                "safety_certified", True
            ),
            lambda value: value["persistence"].__setitem__(
                "boundary", "NO PLATFORM OR NETWORK LOGGING OCCURS"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = json.loads(response.body)["szl_provenance"][
                    "execution_record"
                ]
                mutate(changed)
                changed["record_sha256"] = (
                    verify_execution_record.recompute_record_sha256(changed)
                )
                verified = verify_execution_record.verify_record(changed)
                self.assertTrue(verified["hash_matches"])
                self.assertFalse(verified["semantic_checks_pass"])

    def test_unhandled_compatible_runtime_error_is_shaped_and_not_cached(self):
        original_status = app.state["status"]
        original_llm = app.llm
        try:
            app.state["status"] = "READY"
            app.llm = ExplodingLlm()
            client = TestClient(app.app, raise_server_exceptions=False)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": app.OPENAI_MODEL_ID,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app.state["status"] = original_status
            app.llm = original_llm
        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["code"], "internal_error")
        self.assertNotIn("DO_NOT_LEAK_RUNTIME_DETAIL", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_native_and_openai_routes_share_one_nonblocking_lock(self):
        original_status = app.state["status"]
        original_llm = app.llm
        app.state["status"] = "READY"
        app.llm = FakeLlm()
        acquired = app.inference_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with self.assertRaises(app.HTTPException) as native:
                app.infer(app.InferenceRequest(prompt="hello"))
            with self.assertRaises(app.HTTPException) as compatible:
                app.openai_chat_completions(
                    app.ChatCompletionRequest(
                        model=app.OPENAI_MODEL_ID,
                        messages=[{"role": "user", "content": "hello"}],
                    )
                )
            self.assertEqual(native.exception.status_code, 429)
            self.assertEqual(compatible.exception.status_code, 429)
            self.assertEqual(
                compatible.exception.headers["Retry-After"],
                str(int(app.INFERENCE_BUDGET_SECONDS)),
            )
        finally:
            app.inference_lock.release()
            app.state["status"] = original_status
            app.llm = original_llm

    def test_execution_record_recomputation_golden_vector(self):
        self.assertEqual(
            verify_execution_record.recompute_record_sha256({"b": 1, "a": "two"}),
            "e1e4a2f70c5fb4dad8bb1497da06560b4245ef8ce51b922534aba3f798c6c402",
        )

    def test_openai_error_shape_and_retry_header(self):
        response = app.openai_error_response(
            429,
            "one inference is already running",
            "rate_limit_error",
            "concurrency_limit",
            headers={"Retry-After": "45"},
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(payload["error"]["type"], "rate_limit_error")
        self.assertEqual(payload["error"]["code"], "concurrency_limit")
        self.assertEqual(response.headers["retry-after"], "45")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_release_manifest_covers_execution_verifier(self):
        manifest = app.load_release_manifest()
        self.assertIn("verify_execution_record.py", manifest["source_files"])

    def test_root_waits_for_readiness_before_enabling_inference(self):
        html = app.index()
        self.assertIn('id="run" disabled aria-disabled="true"', html)
        self.assertIn("Checking the runtime and evidence threads", html)
        self.assertIn("getJson('/health')", html)
        self.assertIn("getJson('/version')", html)
        self.assertIn("getJson('/evidence')", html)
        self.assertIn("status==='READY'", html)
        self.assertIn("status==='STARTING'", html)
        self.assertIn("status==='FAILED'", html)
        self.assertIn("if(running)return;if(!hasResult)requestState", html)
        self.assertIn("if(!hasResult)o.textContent", html)
        self.assertIn('data-screenshot-ready="false"', html)
        self.assertIn("prefers-reduced-motion:reduce", html)
        self.assertIn("Every token leaves a", html)
        self.assertIn('name="description"', html)
        self.assertIn('property="og:title"', html)
        self.assertIn('rel="canonical"', html)

    def test_body_limiter_rejects_chunked_overflow(self):
        downstream_called = False
        sent = []
        messages = iter(
            [
                {"type": "http.request", "body": b"abcd", "more_body": True},
                {"type": "http.request", "body": b"ef", "more_body": False},
            ]
        )

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "headers": []}
        asyncio.run(app.BodyLimitMiddleware(downstream, max_bytes=5)(scope, receive, send))
        self.assertFalse(downstream_called)
        self.assertEqual(sent[0]["status"], 413)

    def test_body_limiter_replays_valid_chunked_body(self):
        observed = []
        messages = iter(
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ]
        )

        async def downstream(_scope, receive, _send):
            observed.append((await receive())["body"])

        async def receive():
            return next(messages)

        async def send(_message):
            return None

        scope = {"type": "http", "method": "POST", "headers": []}
        asyncio.run(app.BodyLimitMiddleware(downstream, max_bytes=5)(scope, receive, send))
        self.assertEqual(observed, [b"abcd"])

    def test_body_limiter_uses_openai_error_shape_on_compatible_route(self):
        sent = []

        async def downstream(_scope, _receive, _send):
            self.fail("oversized body must not reach the application")

        async def receive():
            return {"type": "http.request", "body": b"abcdef", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
        asyncio.run(app.BodyLimitMiddleware(downstream, max_bytes=5)(scope, receive, send))
        payload = json.loads(sent[1]["body"])
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["code"], "request_body_error")

    def test_body_limiter_has_absolute_read_timeout(self):
        downstream_called = False
        sent = []

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        async def receive():
            await asyncio.sleep(0.02)
            return {"type": "http.request", "body": b"x", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "headers": []}
        limiter = app.BodyLimitMiddleware(
            downstream, max_bytes=5, read_timeout_seconds=0.001
        )
        asyncio.run(limiter(scope, receive, send))
        self.assertFalse(downstream_called)
        self.assertEqual(sent[0]["status"], 408)


if __name__ == "__main__":
    unittest.main()
