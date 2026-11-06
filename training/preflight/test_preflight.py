"""Preflight unit tests using fake Hub clients - no network, no GPU, no writes."""
import json, os, sys, unittest
sys.path.insert(0, os.path.dirname(__file__))
from unittest import mock
import preflight
from preflight import PreflightError, run_preflight


class FakeFile:
    def __init__(self, rfilename): self.rfilename = rfilename


class FakeApi:
    def __init__(self, files): self._files = files
    def list_repo_files(self, repo, repo_type="model"):
        if "UNREADABLE" in repo: raise RuntimeError("404")
        return [FakeFile(f) for f in self._files]
    def model_info(self, mid):
        if "UNREADABLE" in mid: raise RuntimeError("404")
        return type("I", (), {"sha": "abc123"})()
    def dataset_info(self, did, files_metadata=False):
        if "UNREADABLE" in did: raise RuntimeError("404")
        return type("I", (), {"sha": "def456", "siblings": [FakeFile("data.jsonl")]})()


def write_cfg(**over):
    cfg = {"target_repo": "SZLHOLDINGS/ok-target", "base_model": "SZLHOLDINGS/base",
           "datasets": ["SZLHOLDINGS/ds"], "method": "qlora"}
    cfg.update(over)
    path = "/tmp/test-preflight-cfg.json"
    json.dump(cfg, open(path, "w"))
    return path


class TestPreflight(unittest.TestCase):
    def setUp(self):
        os.environ["HF_TOKEN"] = "test"

    def test_pass(self):
        api = FakeApi(["README.md", "config.json"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            r = run_preflight(write_cfg())
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["base_model_pin"]["sha"], "abc123")

    def test_npz_target_rejected(self):
        api = FakeApi(["README.md", "tiny_khipu.npz"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as cm:
                run_preflight(write_cfg(target_repo="SZLHOLDINGS/npz-target"))
        self.assertEqual(cm.exception.code, "PREFLIGHT_NON_LM_TARGET")

    def test_unreadable_target_rejected(self):
        api = FakeApi([])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as cm:
                run_preflight(write_cfg(target_repo="SZLHOLDINGS/UNREADABLE"))
        self.assertEqual(cm.exception.code, "PREFLIGHT_TARGET_UNREADABLE")

    def test_unreadable_model_rejected(self):
        api = FakeApi(["README.md"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as cm:
                run_preflight(write_cfg(base_model="SZLHOLDINGS/UNREADABLE"))
        self.assertEqual(cm.exception.code, "PREFLIGHT_MODEL_UNREADABLE")

    def test_missing_config_field_rejected(self):
        with self.assertRaises(PreflightError) as cm:
            run_preflight(write_cfg_target_only()
                          if False else _cfg_without("datasets"))
        self.assertEqual(cm.exception.code, "PREFLIGHT_CONFIG_MISSING_FIELD")

    def test_special_method_rejected(self):
        api = FakeApi(["README.md"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as cm:
                run_preflight(write_cfg(method="special-trainer"))
        self.assertEqual(cm.exception.code, "PREFLIGHT_METHOD_NOT_TRAINABLE")


def _cfg_without(field):
    cfg = {"target_repo": "SZLHOLDINGS/ok", "base_model": "SZLHOLDINGS/base",
           "datasets": ["SZLHOLDINGS/ds"], "method": "qlora"}
    del cfg[field]
    path = "/tmp/test-preflight-cfg.json"
    json.dump(cfg, open(path, "w"))
    return path


def write_cfg_target_only():
    raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
