import json, hashlib, pytest
from szl_evolve import croissant_rai as cr

META = {"name": "t", "description": "d", "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "url": "https://huggingface.co/datasets/SZLHOLDINGS/t", "revision": "a" * 40,
        "rai": {"personalSensitiveInformation": "None known; scanned with preupload_check", "dataLimitations": "synthetic"}}

def mkfile(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"id": 1, "score": 0.5, "ok": True, "messages": [{"role": "user", "content": "hi"}]}) + "\n" + json.dumps({"id": 2, "score": 1.5, "ok": False, "messages": []}) + "\n")
    return p

def test_hash_rows_fields(tmp_path):
    p = mkfile(tmp_path); d = cr.build(str(p), META)
    assert d["distribution"][0]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert d["_szl_provenance"]["rows"] == 2
    types = {f["name"].split("/")[1]: f["dataType"] for f in d["recordSet"][0]["field"]}
    assert types == {"id": "sc:Integer", "score": "sc:Float", "ok": "sc:Boolean", "messages": "sc:Text"}

def test_rai_only_from_supplied_never_invented(tmp_path):
    d = cr.build(str(mkfile(tmp_path)), META)
    assert "rai:personalSensitiveInformation" in d and "rai:dataBiases" not in d
    assert "dataBiases" in d["_szl_provenance"]["missing_rai"] and "dataCollection" in d["_szl_provenance"]["missing_rai"]
    assert "http://mlcommons.org/croissant/RAI/1.0" in d["conformsTo"]

def test_strict_check_flags_unpinned(tmp_path):
    m = dict(META); m["revision"] = "main"
    assert any("UNPINNED" in e for e in cr.strict_check(cr.build(str(mkfile(tmp_path)), m)))
    assert cr.strict_check(cr.build(str(mkfile(tmp_path)), META)) == []

def test_validates_with_mlcroissant(tmp_path):
    mlc = pytest.importorskip("mlcroissant")
    out = tmp_path / "c.json"; out.write_text(json.dumps(cr.build(str(mkfile(tmp_path)), META)))
    ds = mlc.Dataset(jsonld=str(out))
    assert ds.metadata.issues.report() == ""
