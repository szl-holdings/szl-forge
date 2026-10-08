# Stage-2 receipts-directory verification harness (run from ROOT): PYTHONUTF8=1 py -3.12 -B logs/v2/stage2_verify_receipts.py
import hashlib, json, os, sys
sys.path.insert(0, os.getcwd())
from v2.research import search
d = "runs/receipts/v2-registered-search"
ob = search.ouroboros()
errs = ob.verify_run_dir(d)
print("ouroboros_adapter", ob.ADAPTER_VERSION, "verify_run_dir(%s)" % d)
print("errors:", len(errs), errs)
files = sorted(os.listdir(d))
print("files:", len(files))
for f in files:
    print("  %s  %s" % (hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest(), f))
dig = search.directory_digest(d)["sha256"]
trace = json.load(open("v2/results/search_trace.json", encoding="utf-8"))
print("directory digest:", dig)
print("trace dir digest:", trace["receipts"]["dir_digest"]["sha256"], "equal:", dig == trace["receipts"]["dir_digest"]["sha256"])
print("search_trace.json sha256:", hashlib.sha256(open("v2/results/search_trace.json", "rb").read()).hexdigest())
import receipt_adapter as ra
schema, src = ra.load_spec_schema()
print("receipt schema source:", src)
recs = [ra.read_receipt(os.path.join(d, f)) for f in files if f != "trace.json"]
print("chain errors:", ra.verify_chain(recs))
print("per-receipt schema/digest/envelope errors:", sum(len(ra.verify_receipt(r, schema=schema)) for r in recs))
print("seq:", [r["seq"] for r in recs], "actions:", sorted(set(r["action"] for r in recs)))
print("cand files:")
for t in trace["trials"]:
    r = t["record"]
    p = "v2/results/search_candidates/%s.model.json" % r["trial_id"]
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("  %s %s match=%s" % (r["trial_id"], h, h == r["model_sha256"]))
