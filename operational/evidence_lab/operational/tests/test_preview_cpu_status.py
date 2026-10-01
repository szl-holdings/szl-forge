"""Check that status and the rendered preview identify CPU execution truthfully."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import external_retrieval_preview as preview


class ExecutionDeviceStatusTests(unittest.TestCase):
    def summary(self, runtime):
        document = {"id": "doc", "title": "Fixture", "text": "The answer is blue."}
        service = SimpleNamespace(
            freeze_sha="a" * 64, run_id="synthetic-device-run", threshold=2.0,
            documents=[document], lock=threading.Lock(),
            payload={"calibration": [], "hotpot": [], "evaluation": [
                {"id": "yes", "question": "Which color?", "context_id": "doc"},
                {"id": "no", "question": "Which city?", "context_id": "doc"}]},
        )
        result = {
            "freeze_sha256": service.freeze_sha,
            "status": "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION",
            "index_sha256": "b" * 64, "calibration_sha256": "c" * 64,
            "runtime": runtime, "completed_at": "SYNTHETIC", "limitations": [],
            "squad_retrieval": {}, "hotpot_support_retrieval": {},
            "given_context_answerability": {}, "retrieved_context_qa_answerable_only": {},
            "records": {"given_context": [
                {"id": "yes", "answerable": True, "margin": 3.0,
                 "prediction": "blue", "gold": ["blue"]},
                {"id": "no", "answerable": False, "margin": 1.0,
                 "prediction": "blue", "gold": []}]},
        }
        hashes = {"documents.npy": "b" * 64, "calibration.json": "c" * 64}
        with mock.patch.object(Path, "read_bytes", return_value=json.dumps(result).encode()), \
                mock.patch.object(preview, "digest", side_effect=lambda path: hashes[path.name]):
            summary = preview.run_summary(service, Path("synthetic-never-read"))
        return service, summary

    def test_cpu_status_preserves_null_gpu_and_names_cpu_execution(self):
        service, summary = self.summary({"device": "cpu", "gpu": None})
        with TestClient(preview.create_preview(service, summary),
                        base_url="http://127.0.0.1:8766") as client:
            response = client.get("/api/status")
        self.assertEqual(200, response.status_code)
        self.assertEqual("cpu", response.json()["execution_device"])
        self.assertIsNone(response.json()["device"])

    def test_cuda_and_legacy_summary_preserve_recorded_hardware(self):
        for runtime, expected in (({"device": "cuda:0", "gpu": "Fixture GPU"}, "cuda:0"),
                                  ({"gpu": "Legacy GPU"}, "unknown")):
            with self.subTest(runtime=runtime):
                _, summary = self.summary(runtime)
                self.assertEqual(expected, summary["execution_device"])
                self.assertEqual(runtime["gpu"], summary["device"])

    def test_busy_cpu_request_uses_device_neutral_message(self):
        service, summary = self.summary({"device": "cpu", "gpu": None})
        service.lock.acquire()
        try:
            with TestClient(preview.create_preview(service, summary),
                            base_url="http://127.0.0.1:8766") as client:
                response = client.post("/api/query", json={"question": "Which color?"},
                                       headers={"X-SZL-Preview": "1"})
        finally:
            service.lock.release()
        self.assertEqual(429, response.status_code)
        self.assertEqual("Local inference is busy. Wait for the current request to finish.",
                         response.json()["detail"])


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI behavior check")
class ExecutionDeviceUiTests(unittest.TestCase):
    def test_initialization_renders_execution_device_and_neutral_loading(self):
        script = r'''
const fs = require("node:fs"), vm = require("node:vm");
const source = fs.readFileSync(process.argv[1], "utf8");
function element() {
  return {textContent:"", value:"Question?", checked:false, children:[],
    classList:{add(){},toggle(){}}, addEventListener(){},setAttribute(){},
    append(node){this.children.push(node);}, reportValidity(){return true;}};
}
async function run(device) {
  const nodes = new Map();
  const summary = {status:"LOCAL_READY_NOT_PRODUCTION",execution_device:device,
    device:device === "cpu" ? null : "Fixture GPU", documents:1,
    capabilities:{identifier_guard:true}, models:{encoder:"fixture",reader:"fixture"},
    metrics:{squad_retrieval:{hybrid_rrf:{recall_at_10:1}},
      hotpot_support_retrieval:{qwen:{all_support_at_5:1}},
      given_context_answerability:{false_answer_count:0,unanswerable_count:1,
        false_answer_rate:0,false_answer_rate_wilson_ci:[0,1]},
      retrieved_context_qa_answerable_only:{exact_match:1}}};
  const context = vm.createContext({
    document:{getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},
      createElement(){return element();}},
    fetch:async (url)=>url === "/api/status" ? {ok:true,json:async()=>summary} : new Promise(()=>{}),
    AbortSignal:{timeout(){return null;}},AbortController,
    setTimeout(){return 1;},clearTimeout(){},URL
  });
  vm.runInContext(source, context);
  await new Promise(resolve=>setImmediate(resolve));
  const health = nodes.get("health").textContent;
  const provenance = nodes.get("provenance").children.map(node=>node.textContent);
  vm.runInContext("submit()", context);
  return {device,health,provenance,busy:nodes.get("request-status").textContent};
}
(async()=>{console.log(JSON.stringify(await Promise.all(["cpu","cuda","cuda:0","unknown"].map(run))));})()
  .catch(error=>{console.error(error);process.exitCode=1;});
'''
        process = subprocess.run(
            [shutil.which("node"), "-e", script,
             str(preview.HERE / "retrieval_web" / "app.js")],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        for result in json.loads(process.stdout):
            with self.subTest(device=result["device"]):
                label = {"cpu": "CPU", "cuda": "GPU", "cuda:0": "GPU", "unknown": "Runtime"}[result["device"]]
                self.assertEqual(f"{label} ready · local", result["health"])
                self.assertIn("Execution device", result["provenance"])
                self.assertIn(result["device"], result["provenance"])
                self.assertEqual("Reading the evidence locally…", result["busy"])


if __name__ == "__main__":
    unittest.main()
