"""Offline controller tests: fake generation exists only in this test module."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "receiptagent_output_server", ROOT / "receiptagent/serve_receiptagent.py"
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def draft():
    return {
        "proposalId": "offline-test",
        "capabilityProfile": "SZL-Forge-1.5B-ReceiptAgent",
        "provenance": "MODEL_PROPOSED",
        "claim": "Draft for human review.",
        "decision": "DRAFT",
        "evidence": [
            {"sourceEndpoint": "/provided/status", "label": "REPORTED", "note": "Caller supplied."}
        ],
        "approvalRequired": True,
        "controllerBoundary": "Human review outside the model.",
        "receiptBinding": {
            "algorithm": "ed25519-over-sha256", "status": "NOT_BOUND", "note": "No receipt."
        },
        "executed": False,
    }


ALLOWLIST = [{"sourceEndpoint": "/provided/status", "label": "REPORTED"}]


def refusal():
    value = draft()
    del value["claim"]
    del value["evidence"]
    value["decision"] = "REFUSE"
    value["reason"] = "Insufficient evidence for this draft."
    return value


class AcceptanceTests(unittest.TestCase):
    def accept(self, value):
        return server.accept_output(json.dumps(value), copy.deepcopy(ALLOWLIST))

    def test_valid_draft_is_preserved_not_repaired(self):
        self.assertEqual(self.accept(draft()), draft())

    def test_structured_refusal_is_preserved(self):
        self.assertEqual(self.accept(refusal()), refusal())

    def test_strict_json_rejects_malformed_trailing_echo_and_nonfinite(self):
        valid = json.dumps(draft())
        values = [
            "", "{", "REFUSE: no evidence", "```json\n" + valid + "\n```",
            "user prompt\n" + valid, valid + "\nexplanation", valid + valid,
            "[]", "null", "true", '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}',
            valid.replace('"offline-test"', '"\\ud800"'),
        ]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(server.GateError):
                    server.accept_output(value, ALLOWLIST)

    def test_duplicate_keys_at_any_depth_are_rejected(self):
        valid = json.dumps(draft())
        values = [
            valid.replace('"decision": "DRAFT"', '"decision": "FINAL", "decision": "DRAFT"'),
            valid.replace('"status": "NOT_BOUND"', '"status": "BOUND", "status": "NOT_BOUND"'),
            valid.replace('"label": "REPORTED"', '"label": "MEASURED", "label": "REPORTED"'),
        ]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(server.GateError):
                    server.accept_output(value, ALLOWLIST)

    def test_parser_rejects_overflow_numeric_literals_before_schema_validation(self):
        for value in (
            '{"value": 1e309}', '{"value": -1e309}',
            '{"nested": [{"value": 1e309}]}',
        ):
            with self.subTest(value=value):
                with self.assertRaises(server.GateError):
                    server.strict_json_object(value)

    def test_missing_forged_and_upgraded_evidence_rejected(self):
        for evidence in (
            [], [{"sourceEndpoint": "/invented", "label": "REPORTED", "note": "Forged."}],
            [{"sourceEndpoint": "/provided/status", "label": "MEASURED", "note": "Upgrade."}],
        ):
            value = draft()
            value["evidence"] = evidence
            with self.subTest(evidence=evidence):
                with self.assertRaises(server.GateError):
                    self.accept(value)
        value = draft()
        del value["evidence"]
        with self.assertRaises(server.GateError):
            self.accept(value)

    def test_authority_inflation_rejected(self):
        changes = {
            "provenance": "SYNTHETIC", "decision": "FINAL", "approvalRequired": False,
            "executed": True, "capabilityProfile": "ReceiptAgent-v2",
            "signature": "invented", "authority": "APPROVED", "receipt": {},
        }
        for key, value in changes.items():
            candidate = draft()
            candidate[key] = value
            with self.subTest(key=key):
                with self.assertRaises(server.GateError):
                    self.accept(candidate)
        for key, value in (("status", "BOUND"), ("signature", "invented")):
            candidate = draft()
            candidate["receiptBinding"][key] = value
            with self.assertRaises(server.GateError):
                self.accept(candidate)
        candidate = draft()
        candidate["evidence"][0]["signature"] = "invented"
        with self.assertRaises(server.GateError):
            self.accept(candidate)

    def test_refusal_cannot_carry_draft_authority_or_extra_evidence(self):
        for key, value in (("executed", True), ("signature", "invented"), ("evidence", [])):
            candidate = refusal()
            candidate[key] = value
            with self.subTest(key=key):
                with self.assertRaises(server.GateError):
                    self.accept(candidate)

    def test_missing_or_malformed_allowlist_rejected_even_for_refusal(self):
        for value in (
            None, [], {}, ["citation"], [{"sourceEndpoint": "/provided/status"}],
            [{"sourceEndpoint": "", "label": "REPORTED"}],
            [{"sourceEndpoint": "/provided/status", "label": "CERTIFIED"}],
            [{"sourceEndpoint": "/provided/status", "label": "REPORTED", "signature": "x"}],
            ALLOWLIST + [{"sourceEndpoint": "/provided/status", "label": "MEASURED"}],
        ):
            with self.subTest(value=value):
                with self.assertRaises(server.GateError):
                    server.accept_output(json.dumps(refusal()), value)


class RequestBoundaryTests(unittest.TestCase):
    def request(self, **updates):
        value = {"prompt": "Draft from supplied evidence.", "evidence_allowlist": ALLOWLIST}
        value.update(updates)
        return value

    def test_request_validation_precedes_generation(self):
        for value in (
            {"prompt": "no allowlist"}, self.request(evidence_allowlist=[]),
            self.request(prompt=""), self.request(prompt=12), self.request(max_new_tokens=True),
            self.request(max_new_tokens=0), self.request(max_new_tokens=257),
            self.request(approved=True),
        ):
            with self.subTest(value=value), mock.patch.object(server, "generate") as generate:
                with self.assertRaises(server.GateError):
                    server.propose(value)
                generate.assert_not_called()

    def test_fake_generation_only_in_test_and_no_raw_text_success(self):
        with mock.patch.object(server, "availability", return_value={"status": "READY"}):
            with mock.patch.object(server, "generate", return_value=json.dumps(draft())) as generate:
                value = server.propose(self.request())
        generate.assert_called_once_with("Draft from supplied evidence.", 256)
        self.assertEqual(value["result"], draft())
        self.assertEqual(value["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertNotIn("text", value)
        with mock.patch.object(server, "availability", return_value={"status": "READY"}):
            with mock.patch.object(server, "generate", return_value="unsafe arbitrary text"):
                with self.assertRaises(server.GateError):
                    server.propose(self.request())

    def test_handler_rejects_duplicate_request_before_generation(self):
        handler = object.__new__(server.Handler)
        handler.path = "/"
        raw = b'{"prompt":"first","prompt":"second","evidence_allowlist":[]}'
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler._send = mock.Mock()
        with mock.patch.object(server, "generate") as generate:
            handler.do_POST()
        generate.assert_not_called()
        self.assertEqual(handler._send.call_args.args[0], 400)
        self.assertNotIn("text", handler._send.call_args.args[1])

    def test_handler_returns_explicit_output_rejection(self):
        handler = object.__new__(server.Handler)
        handler.path = "/"
        raw = json.dumps(self.request()).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler._send = mock.Mock()
        with mock.patch.object(server, "availability", return_value={"status": "READY"}):
            with mock.patch.object(server, "generate", return_value=json.dumps({"executed": True})):
                handler.do_POST()
        self.assertEqual(handler._send.call_args.args[0], 422)
        self.assertEqual(handler._send.call_args.args[1]["error"], "OUTPUT_REJECTED")

    def test_no_generation_when_weights_unavailable(self):
        with mock.patch.object(server, "availability", return_value={"status": "UNAVAILABLE"}):
            with mock.patch.object(server, "generate") as generate:
                with self.assertRaises(server.GateError):
                    server.propose(self.request())
        generate.assert_not_called()


class CompletionSlicingTests(unittest.TestCase):
    def test_only_new_tokens_are_decoded_without_loading_a_real_model(self):
        class Tokens(list):
            @property
            def shape(self):
                return (1, 2)

        tokenizer = mock.Mock(return_value={"input_ids": Tokens([[11, 12]])})
        tokenizer.decode.return_value = '{"decision":"REFUSE"}'
        model = mock.Mock()
        model.generate.return_value = [[11, 12, 21, 22]]
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=tokenizer)),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=mock.Mock(return_value=model)),
        )
        torch = types.SimpleNamespace(no_grad=mock.MagicMock())
        with mock.patch.dict(sys.modules, {"transformers": transformers, "torch": torch}):
            self.assertEqual(server.generate("private prompt", 8), '{"decision":"REFUSE"}')
        tokenizer.decode.assert_called_once_with([21, 22], skip_special_tokens=True)
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
