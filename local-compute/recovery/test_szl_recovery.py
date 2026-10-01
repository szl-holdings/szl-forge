# SPDX-License-Identifier: Apache-2.0
"""Offline contracts. These tests do not access an owner GPU or local Ollama."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from contextlib import ExitStack
from urllib.error import HTTPError
import szl_recovery as m


def response(case, text=None, reason="stop"):
    return {"model": m.RECEIPT, "message": {"role":"assistant", "content": json.dumps(case["expected"]) if text is None else text},
            "done":True, "done_reason":reason, "eval_count":10,"load_duration":10,"eval_duration":10}


def healthy_gpu():
    return {"temperature_c":55,"free_mib":6500,"compute_processes":[]}


class FakeClient:
    def __init__(self, wrong_schema=False, fail_at=None):
        self.requests=[]; self.wrong_schema=wrong_schema; self.fail_at=fail_at
    def call(self,path,payload=None,timeout=15):
        if path=="/api/ps": return {"models":[]}
        if path=="/api/tags": return {"models":[{"name":m.RECEIPT,"digest":m.EXPECTED[m.RECEIPT]}]}
        if path=="/api/chat":
            if self.fail_at == len(self.requests): raise TimeoutError("secret-shaped-marker-must-not-be-exported")
            self.requests.append(copy.deepcopy(payload))
            case=next(c for c in m.CASES if c["prompt"]==payload["messages"][1]["content"])
            value=copy.deepcopy(case["expected"])
            if case["id"]=="draft_only" and ("format" not in payload or self.wrong_schema):value["claim"]="extra"
            return response(case,json.dumps(value))
        raise AssertionError(path)


class Tests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(m.GateError):m.strict_json('{"x":1,"x":2}')
    def test_nonfinite_numbers_rejected(self):
        for value in ("NaN","Infinity","-Infinity","1e9999"):
            with self.subTest(value=value), self.assertRaises(m.GateError):m.strict_json(value)
    def test_large_json_rejected(self):
        with self.assertRaises(m.GateError):m.strict_json(" "*(8*1024**2+1))
    def test_boolean_is_not_integer(self):
        self.assertFalse(m.conforms(True,{"type":"integer"}));self.assertFalse(m.same_typed(True,1))
    def test_float_is_not_integer(self):
        self.assertFalse(m.conforms(1.0,{"type":"integer"}))
    def test_observed_receipt_extra_claim_remains_failure(self):
        case=m.CASES[3]
        r=response(case,'{"claim":"Produce a text-only draft marker.","executed":false,"approval_required":true}')
        v=m.assess(case,r)
        self.assertTrue(v["normal_stop"]);self.assertFalse(v["json_schema_valid"]);self.assertFalse(v["passed"])
        self.assertIn('"claim"',v["response"])
    def test_correct_draft_passes(self):
        self.assertTrue(m.assess(m.CASES[3],response(m.CASES[3]))["passed"])
    def test_schema_valid_but_false_safety_flags_fail_semantics(self):
        v=m.assess(m.CASES[3],response(m.CASES[3],'{"executed":true,"approval_required":false}'))
        self.assertTrue(v["json_schema_valid"]);self.assertFalse(v["passed"])
    def test_correct_answer_at_token_limit_does_not_pass(self):
        self.assertFalse(m.assess(m.CASES[3],response(m.CASES[3],reason="length"))["passed"])
    def test_repeated_at_signs_fail(self):
        v=m.assess(m.CASES[0],response(m.CASES[0],"@"*128,"length"))
        self.assertFalse(v["passed"]);self.assertFalse(v["normal_stop"])
    def test_duplicate_response_keys_do_not_pass(self):
        text='{"executed":true,"executed":false,"approval_required":true}'
        self.assertFalse(m.assess(m.CASES[3],response(m.CASES[3],text))["passed"])
    def test_tool_call_refused(self):
        r=response(m.CASES[0]);r["message"]["tool_calls"]=[{"name":"do-something"}]
        with self.assertRaises(m.GateError):m.assess(m.CASES[0],r)
    def test_other_model_response_refused(self):
        r=response(m.CASES[0]);r["model"]="szl1:latest"
        with self.assertRaises(m.GateError):m.assess(m.CASES[0],r)
    def test_markdown_not_stripped(self):
        self.assertFalse(m.assess(m.CASES[1],response(m.CASES[1],'```json\n[-3,2,8]\n```'))["passed"])
    def test_arithmetic_exact_text_unchanged(self):
        self.assertFalse(m.assess(m.CASES[0],response(m.CASES[0],'1.02e2'))["passed"])
    def test_nested_array_types_checked(self):
        self.assertFalse(m.conforms([True,2],m.SCHEMAS["ordering"]))
    def test_all_expected_shapes_valid(self):
        for c in m.CASES:self.assertTrue(m.conforms(c["expected"],m.SCHEMAS[c["id"]]))
    def test_no_answer_values_in_schemas(self):
        def walk(obj):
            if isinstance(obj,dict):
                self.assertFalse(set(obj)&{"const","enum","default","examples","minimum","maximum","minItems","maxItems"})
                for x in obj.values():walk(x)
            elif isinstance(obj,list):
                for x in obj:walk(x)
        walk(m.SCHEMAS)
        self.assertTrue(m.conforms({"executed":True,"approval_required":False},m.SCHEMAS["draft_only"]))
    def test_schema_and_control_requests_have_identical_prompts(self):
        for c in m.CASES:
            a=m.build_request(c,False);b=m.build_request(c,True)
            self.assertEqual(a,{k:v for k,v in b.items() if k!="format"})
    def test_no_tools_in_request(self):
        self.assertNotIn("tools",m.build_request(m.CASES[0],True))
    def test_only_last_request_unloads(self):
        self.assertEqual(m.build_request(m.CASES[0],True)["keep_alive"],"30s")
        self.assertEqual(m.build_request(m.CASES[0],True,True)["keep_alive"],0)
    def test_mutation_and_remote_endpoint_refused(self):
        for path in ("/api/pull","/api/create","/api/delete","/api/generate","https://example.com/api/chat"):
            with self.subTest(path=path),self.assertRaises(m.GateError):m.LocalClient.validate_request(path,{})
    def test_bad_models_cannot_be_run(self):
        for name in m.BLOCKED:
            p=m.build_request(m.CASES[0],True);p["model"]=name
            with self.assertRaises(m.GateError):m.LocalClient.validate_request("/api/chat",p)
    def test_request_settings_not_coerced(self):
        p=m.build_request(m.CASES[0],True);p["options"]["temperature"]=False
        with self.assertRaises(m.GateError):m.LocalClient.validate_request("/api/chat",p)
    def test_expected_requests_admitted(self):
        for c in m.CASES:
            for mode in (True,False):m.LocalClient.validate_request("/api/chat",m.build_request(c,mode))
    def test_redirects_refused(self):
        with self.assertRaises(m.GateError):m.NoRedirects().redirect_request(None)
    def test_identity_pin_cannot_drift(self):
        meta={"details":{"format":"gguf"},"capabilities":["completion"]}
        with self.assertRaises(m.GateError):m.admit_receipt({"digest":"a"*64,"size":100},meta)
    def test_cloud_metadata_refused(self):
        meta={"details":{"format":"gguf"},"capabilities":["completion"],"remote_host":"cloud"}
        with self.assertRaises(m.GateError):m.admit_receipt({"digest":m.EXPECTED[m.RECEIPT],"size":100},meta)
    def test_matching_local_model_admitted(self):
        m.admit_receipt({"digest":m.EXPECTED[m.RECEIPT],"size":100},{"details":{"format":"gguf"},"capabilities":["completion"]})
    def test_gpu_busy_refuses_without_stopping_process(self):
        v=healthy_gpu();v["compute_processes"]=[{"pid":1,"executable_name":"python.exe"}]
        with self.assertRaises(m.GateError):m.require_idle_gpu(v)
    def test_memory_and_temperature_limits(self):
        for key,value in (("free_mib",100),("temperature_c",80)):
            v=healthy_gpu();v[key]=value
            with self.assertRaises(m.GateError):m.require_idle_gpu(v)
    def test_metadata_redacts_prompt_and_local_path(self):
        model={"modelfile":'FROM C:\\private\\blobs\\sha256-'+'a'*64+"\n", "system":"secret-system-text","template":"template"}
        v=m.metadata_projection("szl1:latest",{"digest":m.EXPECTED["szl1:latest"]},model)
        self.assertEqual(v["provider_reported_from_blob_sha256"],"a"*64)
        self.assertNotIn("private",json.dumps(v));self.assertNotIn("secret-system-text",json.dumps(v))
        self.assertFalse(v["blob_bytes_independently_verified"])
    def test_timeout_error_has_no_sensitive_body(self):
        self.assertNotIn("secret",json.dumps(m.safe_error(TimeoutError("secret-token"))))
    def test_http_error_retains_status_not_body(self):
        v=m.safe_error(HTTPError("http://secret",500,"sensitive",{},None))
        self.assertEqual(v["http_status"],500);self.assertNotIn("sensitive",json.dumps(v))
    def test_new_file_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"x.json";m.write_new(p,{"old":True})
            with self.assertRaises(FileExistsError):m.write_new(p,{"new":True})
            self.assertEqual(json.loads(p.read_text()),{"old":True})
    def test_baseline_files_are_pinned(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            content={"summary.json":json.dumps({"state":"PROTOCOLS_COMPLETED","source_revision":m.FORGE_SHA}).encode()}
            for name,raw in content.items():(root/name).write_bytes(raw)
            pins={n:m.sha(b) for n,b in content.items()}
            with patch.dict(m.EVIDENCE_HASHES,pins,clear=True):
                self.assertEqual(m.verify_evidence(root),pins)
                (root/"summary.json").write_text('{}')
                with self.assertRaises(m.GateError):m.verify_evidence(root)
    def test_paired_experiment_retains_original_5_of_6(self):
        with tempfile.TemporaryDirectory() as t:
            r={};c=FakeClient();m.paired_check(c,Path(t),r,gpu_probe=healthy_gpu)
            self.assertEqual(len(c.requests),12)
            self.assertEqual(r["scores"]["unconstrained"]["passed"],5)
            self.assertEqual(r["scores"]["schema"]["passed"],6)
            self.assertFalse(r["publication_eligible"])
            self.assertEqual(r["original_unconstrained_score"],"5/6_UNCHANGED")
    def test_schema_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as t:
            r={};m.paired_check(FakeClient(wrong_schema=True),Path(t),r,gpu_probe=healthy_gpu)
            self.assertEqual(r["state"],"SCHEMA_INTERFACE_NOT_QUALIFIED")
    def test_timeout_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as t:
            c=FakeClient(fail_at=1);r={}
            with self.assertRaises(TimeoutError):m.paired_check(c,Path(t),r,gpu_probe=healthy_gpu)
            self.assertEqual(len(c.requests),1)
    def test_input_paths_not_evaluated_as_commands(self):
        v=m.metadata_projection("szl1:latest",{}, {"modelfile":"FROM $(launch-something)\n"})
        self.assertIsNone(v["provider_reported_from_blob_sha256"])


    def test_foreign_lock_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root=Path(t)
            stack.enter_context(patch.object(m.platform,"system",return_value="Windows"))
            stack.enter_context(patch.object(m.platform,"node",return_value="BETTERWITHAGE"))
            stack.enter_context(patch.object(m.sys,"version_info",(3,12,10)))
            stack.enter_context(patch.object(m.sys,"argv",["recovery"]))
            stack.enter_context(patch.object(m,"verify_package"))
            stack.enter_context(patch.object(m.Path,"home",return_value=root))
            lock=root/".szl-two-machine-lab.lock";lock.write_text('foreign owner')
            code=m.main()
            self.assertEqual(code,1);self.assertEqual(lock.read_text(),'foreign owner')
            report=json.loads(next(root.glob('szl-recovery-*/recovery-report.json')).read_text())
            self.assertEqual(report['error_code'],'EXISTING_LAB_LOCK_NO_DUPLICATE_JOB')
            self.assertTrue(next(root.glob('szl-recovery-*/RETURN_TO_CHAT.zip')).is_file())
    def test_unsupported_host_refused_before_metadata(self):
        with patch.object(m.platform,"system",return_value="Linux"), patch.object(m.sys,"argv",["recovery"]), patch.object(m,"LocalClient") as client:
            with self.assertRaisesRegex(m.GateError,"BETTERWITHAGE"):m.main()
            client.assert_not_called()
    def test_loaded_foreign_model_refused(self):
        c=FakeClient()
        base=c.call
        def call(path,payload=None,timeout=15):
            if path=='/api/ps':return {'models':[{'name':'other:latest','digest':'b'*64}]}
            return base(path,payload,timeout)
        c.call=call
        with self.assertRaises(m.GateError):m.guarded_live_identity(c,allowed_loaded=False)
        with self.assertRaises(m.GateError):m.guarded_live_identity(c,allowed_loaded=True)


    def test_inspection_is_default(self):
        self.assertFalse(m.parse_arguments([]).run_paired)

    def test_inference_requires_explicit_opt_in(self):
        self.assertTrue(m.parse_arguments(["--run-paired"]).run_paired)
        self.assertFalse(m.parse_arguments(["--inspect-only"]).run_paired)

    def test_conflicting_modes_fail_before_execution(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
            m.parse_arguments(["--inspect-only", "--run-paired"])
        self.assertEqual(failure.exception.code, 2)

    def test_repo_package_integrity(self):
        m.verify_package()

    def test_explicit_inspection_does_not_call_paired_runner(self):
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root = Path(t)
            stack.enter_context(patch.object(m.platform, "system", return_value="Windows"))
            stack.enter_context(patch.object(m.platform, "node", return_value="BETTERWITHAGE"))
            stack.enter_context(patch.object(m.sys, "version_info", (3,12,10)))
            stack.enter_context(patch.object(m.sys, "argv", ["recovery"]))
            stack.enter_context(patch.object(m, "verify_package"))
            stack.enter_context(patch.object(m.Path, "home", return_value=root))
            stack.enter_context(patch.object(m, "verify_evidence", return_value={"fixture": "a"*64}))
            stack.enter_context(patch.object(m, "recipe_presence", return_value={}))
            stack.enter_context(patch.object(m, "gpu_observation", return_value=healthy_gpu()))
            stack.enter_context(patch.object(m, "LocalClient", return_value=Mock()))
            stack.enter_context(patch.object(m, "tags", return_value={}))
            stack.enter_context(patch.object(m, "loaded_models", return_value=[]))
            client = m.LocalClient()
            client.call.return_value = {"version": "test-fixture"}
            paired = stack.enter_context(patch.object(m, "paired_check"))
            code = m.main()
            self.assertEqual(code, 0)
            paired.assert_not_called()
            report = json.loads(next(root.glob("szl-recovery-*/recovery-report.json")).read_text())
            self.assertEqual(report["state"], "METADATA_RECORDED_NO_INFERENCE_REQUESTED")
            self.assertFalse(report["trained"])

if __name__=="__main__":unittest.main()
