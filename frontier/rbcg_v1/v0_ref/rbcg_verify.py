#!/usr/bin/env python3
# RBCG v0 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg_verify.py -- fail-closed offline verifier for RBCG v0 guarantee and monitor receipts.

IN-SILICO RESEARCH ONLY. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE.

Inputs: guarantee envelope, public key, calibration rows; optionally a monitor chain, deployment
rows and labels. Verdicts:
  VALID                 every check passed and no event fired
  VIOLATED              labels arrived and, while the guarantee was in force, the one-sided
                        Clopper-Pearson lower bound on the deployment selective error exceeded alpha
                        at the spent audit level delta_audit * 6 / (pi^2 j^2) of look j
  REFUSED:<CODE>        MISSING_INPUT, SCHEMA, BAD_SIGNATURE, HASH_MISMATCH, RECOMPUTE_MISMATCH,
                        CLAIM_UNSUPPORTED, CHAIN_BROKEN, ORDER, LEAKAGE, SHIFT_ALARM, INTERNAL
Every exception becomes a refusal; nothing raises out of verify(). Floats, NaN, Infinity,
duplicate keys, missing or extra fields and wrong types are refused in every document.
The recomputation of the LTT table and of the monitor is written independently of rbcg.py's
fast paths (direct definitions), so a fast-path bug cannot hide behind a shared code path.
"""
import base64
import hashlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbcg  # noqa: E402

CODES = ("MISSING_INPUT", "SCHEMA", "BAD_SIGNATURE", "HASH_MISMATCH", "RECOMPUTE_MISMATCH",
         "CLAIM_UNSUPPORTED", "CHAIN_BROKEN", "ORDER", "LEAKAGE", "SHIFT_ALARM", "INTERNAL")
DEFAULT_POLICY = {"alpha_e6": 100000, "delta_max_e6": 100000, "delta_s_max_e6": 10000,
                  "delta_audit_e6": 50000, "require_order_commitment": True,
                  "trusted_rbcg_sha256": None, "expected_bindings": {}}
HEX64 = re.compile(r"[0-9a-f]{64}")
LABEL_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
ID_RE = re.compile(r"[A-Za-z0-9._:/|-]{1,200}")
SUPPORTED_GRID_ORDER = "descending"


class Refuse(Exception):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------ strict parsing helpers
def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise Refuse("SCHEMA", f"duplicate key {k!r}")
        d[k] = v
    return d


def _no_float(s):
    raise Refuse("SCHEMA", f"floating-point number {s!r} not allowed (integers only)")


def _no_constant(s):
    raise Refuse("SCHEMA", f"non-finite constant {s!r} not allowed")


def loads_strict(raw, what):
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raise Refuse("SCHEMA", f"{what}: not UTF-8")
    if type(raw) is not str:
        raise Refuse("SCHEMA", f"{what}: expected JSON text, got {type(raw).__name__}")
    try:
        return json.loads(raw, object_pairs_hook=_no_dupes, parse_float=_no_float,
                          parse_constant=_no_constant)
    except Refuse:
        raise
    except Exception as e:
        raise Refuse("SCHEMA", f"{what}: invalid JSON ({type(e).__name__})")


def _obj(d, keys, path):
    if type(d) is not dict:
        raise Refuse("SCHEMA", f"{path} must be an object")
    have, want = set(d), set(keys)
    if have != want:
        raise Refuse("SCHEMA", f"{path}: missing {sorted(want - have)} extra {sorted(have - want)}")


def _int(v, path, lo=None, hi=None):
    if type(v) is not int:
        raise Refuse("SCHEMA", f"{path} must be an integer, got {type(v).__name__}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise Refuse("SCHEMA", f"{path}={v} outside [{lo}, {hi}]")


def _bool(v, path):
    if type(v) is not bool:
        raise Refuse("SCHEMA", f"{path} must be a boolean")


def _str(v, path, pattern=None, maxlen=4000):
    if type(v) is not str or not v or len(v) > maxlen:
        raise Refuse("SCHEMA", f"{path} must be a non-empty string (<= {maxlen} chars)")
    if pattern is not None and not pattern.fullmatch(v):
        raise Refuse("SCHEMA", f"{path} has an invalid format")


def _b64(s, path):
    """DSSE: standard or URL-safe base64 must be accepted; the encoding must be canonical."""
    if type(s) is not str:
        raise Refuse("SCHEMA", f"{path} must be a base64 string")
    for alt in (None, b"-_"):
        try:
            b = base64.b64decode(s, altchars=alt, validate=True)
        except Exception:
            continue
        re_enc = base64.b64encode(b, altchars=alt).decode("ascii")
        if re_enc == s:
            return b
    raise Refuse("SCHEMA", f"{path}: not canonical standard or URL-safe base64")


# ------------------------------------------------------------------ envelopes
def parse_envelope(raw, expected_type, pub, what):
    env = loads_strict(raw, what)
    _obj(env, ("payload", "payloadType", "signatures"), what)
    if env["payloadType"] != expected_type:
        raise Refuse("SCHEMA", f"{what}: payloadType {env['payloadType']!r} is not {expected_type!r}")
    sigs = env["signatures"]
    if type(sigs) is not list or len(sigs) != 1:
        raise Refuse("SCHEMA", f"{what}: exactly one signature required")
    _obj(sigs[0], ("keyid", "sig"), f"{what}.signatures[0]")
    _str(sigs[0]["keyid"], f"{what}.keyid", HEX64)
    body = _b64(env["payload"], f"{what}.payload")
    sig = _b64(sigs[0]["sig"], f"{what}.sig")
    if sigs[0]["keyid"] != pub.keyid():
        raise Refuse("BAD_SIGNATURE", f"{what}: keyid does not match the supplied public key")
    if not pub.verify(rbcg.pae(expected_type, body), sig):
        raise Refuse("BAD_SIGNATURE", f"{what}: ECDSA P-256 signature over the DSSE PAE is invalid")
    payload = loads_strict(body, f"{what}.payload")
    if rbcg.canonical_json(payload) != body:
        raise Refuse("SCHEMA", f"{what}: payload is not canonical JSON")
    return env, payload


def _public_key(pk):
    if pk is None:
        raise Refuse("MISSING_INPUT", "public key not supplied")
    try:
        if isinstance(pk, rbcg.PublicKey):
            return pk
        if type(pk) is str and pk.lstrip().startswith("-----BEGIN"):
            return rbcg.PublicKey.from_pem(pk)
        if type(pk) is str:
            return rbcg.PublicKey.from_sec1_hex(pk.strip())
    except ValueError as e:
        raise Refuse("BAD_SIGNATURE", f"public key invalid: {e}")
    raise Refuse("SCHEMA", "public key must be a PEM or SEC1 hex string")


# ------------------------------------------------------------------ payload schemas
BINDING_KEYS = ("advisory_code_sha256", "governance_code_sha256", "model_id", "model_sha256",
                "rbcg_code_sha256", "sim_code_sha256", "weights_file_sha256")
G_KEYS = ("audit", "bindings", "calibration", "claim", "issued_at_utc", "monitor",
          "not_for_clinical_use", "procedure", "rbcg_version", "schema", "synthetic_only")
CAL_KEYS = ("env_profile", "n_rows", "n_units", "row_schema", "rows_sha256", "seed_family")
PROC_KEYS = ("alpha_e6", "delta_e6", "grid_c", "grid_order", "id", "n_rejected", "p_value",
             "selector", "table_sha256", "tau_star_c")
MON_KEYS = ("betting_start", "delta_s_e6", "epsilons_e3", "martingale_id", "n_cal_units",
            "p_value_id", "score_id", "theta_seed", "unit")
M_KEYS = ("alarm", "batch_n_rows", "batch_sha256", "deploy_source", "guarantee_sha256", "log_m_e9",
          "p_e12", "prev_sha256", "schema", "score", "seq", "unit_id")


def check_guarantee_schema(g):
    _obj(g, G_KEYS, "guarantee")
    if g["schema"] != rbcg.GUARANTEE_SCHEMA:
        raise Refuse("SCHEMA", "guarantee.schema")
    _str(g["rbcg_version"], "rbcg_version")
    _bool(g["synthetic_only"], "synthetic_only")
    _bool(g["not_for_clinical_use"], "not_for_clinical_use")
    if g["synthetic_only"] is not True or g["not_for_clinical_use"] is not True:
        raise Refuse("CLAIM_UNSUPPORTED", "receipt does not declare synthetic-only, non-clinical use")
    _str(g["claim"], "claim")
    if type(g["issued_at_utc"]) is not str:
        raise Refuse("SCHEMA", "issued_at_utc must be a string")
    b = g["bindings"]
    _obj(b, BINDING_KEYS, "bindings")
    for k in BINDING_KEYS:
        _str(b[k], f"bindings.{k}", None if k == "model_id" else HEX64)
    c = g["calibration"]
    _obj(c, CAL_KEYS, "calibration")
    _str(c["seed_family"], "calibration.seed_family", ID_RE)
    _str(c["env_profile"], "calibration.env_profile", ID_RE)
    _int(c["n_rows"], "calibration.n_rows", 1)
    _int(c["n_units"], "calibration.n_units", 1)
    _str(c["rows_sha256"], "calibration.rows_sha256", HEX64)
    if c["row_schema"] != rbcg.ROW_SCHEMA:
        raise Refuse("SCHEMA", "calibration.row_schema")
    p = g["procedure"]
    _obj(p, PROC_KEYS, "procedure")
    _int(p["alpha_e6"], "procedure.alpha_e6", 1, 999999)
    _int(p["delta_e6"], "procedure.delta_e6", 1, 999999)
    if type(p["grid_c"]) is not list or not p["grid_c"]:
        raise Refuse("SCHEMA", "procedure.grid_c must be a non-empty list")
    for i, t in enumerate(p["grid_c"]):
        _int(t, f"procedure.grid_c[{i}]", 1, 100)
    _str(p["grid_order"], "procedure.grid_order")
    _str(p["id"], "procedure.id")
    _str(p["p_value"], "procedure.p_value")
    _obj(p["selector"], ("id", "ood_floor_c"), "procedure.selector")
    _str(p["selector"]["id"], "procedure.selector.id")
    _int(p["selector"]["ood_floor_c"], "procedure.selector.ood_floor_c", 0, 100)
    _str(p["table_sha256"], "procedure.table_sha256", HEX64)
    _int(p["n_rejected"], "procedure.n_rejected", 0)
    if p["tau_star_c"] is not None:
        _int(p["tau_star_c"], "procedure.tau_star_c", 1, 100)
    m = g["monitor"]
    _obj(m, MON_KEYS, "monitor")
    for k in ("betting_start", "martingale_id", "p_value_id", "score_id", "unit"):
        _str(m[k], f"monitor.{k}")
    _int(m["delta_s_e6"], "monitor.delta_s_e6", 1, 999999)
    if type(m["epsilons_e3"]) is not list or not m["epsilons_e3"]:
        raise Refuse("SCHEMA", "monitor.epsilons_e3 must be a non-empty list")
    for i, e in enumerate(m["epsilons_e3"]):
        _int(e, f"monitor.epsilons_e3[{i}]", 1, 999)
    _str(m["theta_seed"], "monitor.theta_seed", HEX64)
    _int(m["n_cal_units"], "monitor.n_cal_units", 1)
    a = g["audit"]
    _obj(a, ("delta_audit_e6", "spending"), "audit")
    _int(a["delta_audit_e6"], "audit.delta_audit_e6", 1, 999999)
    _str(a["spending"], "audit.spending")


def check_monitor_schema(m, path):
    _obj(m, M_KEYS, path)
    if m["schema"] != rbcg.MONITOR_SCHEMA:
        raise Refuse("SCHEMA", f"{path}.schema")
    _bool(m["alarm"], f"{path}.alarm")
    _int(m["batch_n_rows"], f"{path}.batch_n_rows", 1)
    for k in ("batch_sha256", "guarantee_sha256", "prev_sha256"):
        _str(m[k], f"{path}.{k}", HEX64)
    _str(m["deploy_source"], f"{path}.deploy_source", ID_RE)
    _str(m["unit_id"], f"{path}.unit_id", ID_RE)
    _int(m["seq"], f"{path}.seq", 1)
    _int(m["score"], f"{path}.score", 0, rbcg.E12)
    _int(m["p_e12"], f"{path}.p_e12", 0, rbcg.E12)
    _int(m["log_m_e9"], f"{path}.log_m_e9")


def check_rows(rows, keys, what):
    if type(rows) is not list or not rows:
        raise Refuse("SCHEMA", f"{what} must be a non-empty list of row objects")
    seen = set()
    for i, r in enumerate(rows):
        path = f"{what}[{i}]"
        _obj(r, keys, path)
        _str(r["cid"], f"{path}.cid", ID_RE)
        _str(r["unit"], f"{path}.unit", ID_RE)
        _bool(r["inf"], f"{path}.inf")
        _bool(r["ood"], f"{path}.ood")
        _int(r["pmax_e12"], f"{path}.pmax_e12", 0, rbcg.E12)
        _str(r["pred"], f"{path}.pred", LABEL_RE)
        if "label" in keys:
            _str(r["label"], f"{path}.label", LABEL_RE)
        if r["cid"] in seen:
            raise Refuse("SCHEMA", f"{path}.cid duplicated")
        seen.add(r["cid"])


def _units(rows, what):
    try:
        return rbcg.group_units(rows)
    except ValueError as e:
        raise Refuse("SCHEMA", f"{what}: {e}")


# ------------------------------------------------------------------ independent recomputation
def recompute_table(rows, grid, alpha_e6, floor_c):
    """Direct definition, written independently of rbcg.ltt_table's fast path."""
    alpha, out = alpha_e6 / 1e6, []
    for t in grid:
        n = k = 0
        for r in rows:
            if not r["inf"]:
                continue
            thr = (max(t, floor_c) if r["ood"] else t) * 10 ** 10
            if r["pmax_e12"] >= thr:
                n += 1
                if r["pred"] != r["label"]:
                    k += 1
        p = rbcg.binom_cdf(k, n, alpha) if n > 0 else 1.0
        out.append({"k": k, "n": n, "p_e9": int(math.ceil(p * 1e9)), "tau_c": t})
    return out


def recompute_fixed_sequence(table, delta_e6):
    tau, n_rej = None, 0
    for e in table:
        if e["p_e9"] > delta_e6 * 1000:
            break
        tau, n_rej = e["tau_c"], n_rej + 1
    return tau, n_rej


def _theta(seed, j):
    d = hashlib.sha256(f"{seed}|{j}".encode("ascii")).digest()
    return ((int.from_bytes(d[:7], "big") >> 3) + 1) / 2.0 ** 53


def _p_direct(prev_scores, s, th):
    gt = sum(1 for x in prev_scores if x > s)
    eq = sum(1 for x in prev_scores if x == s) + 1
    return (gt + th * eq) / (len(prev_scores) + 1)


def _unit_score(rows):
    return sum(rbcg.E12 - r["pmax_e12"] for r in rows) // len(rows)


def _claim_expected(g):
    p, c, m = g["procedure"], g["calibration"], g["monitor"]
    return rbcg.claim_text(g["bindings"]["model_id"], p["tau_star_c"], p["selector"]["ood_floor_c"],
                           p["alpha_e6"], p["delta_e6"], c["n_rows"], c["seed_family"],
                           m["delta_s_e6"])


# ------------------------------------------------------------------ main entry
def _result(verdict, code=None, detail="", **kw):
    out = {"verdict": verdict, "code": code, "detail": detail, "guarantee_sha256": None,
           "tau_star_c": None, "n_batches": 0, "alarm_batch": None, "violation_batch": None,
           "trace": [], "audit": None}
    out.update(kw)
    return out


def verify(guarantee, public_key, cal_rows, monitor_chain=None, deploy_rows=None, labels=None,
           policy=None):
    """Never raises. Returns a dict with 'verdict' in {VALID, VIOLATED, REFUSED:<CODE>}."""
    st = {"guarantee_sha256": None, "tau_star_c": None, "n_batches": 0, "alarm_batch": None,
          "violation_batch": None, "trace": [], "audit": None}
    try:
        verdict, detail = _verify(st, guarantee, public_key, cal_rows, monitor_chain, deploy_rows,
                                  labels, policy)
        return _result(verdict, None, detail, **st)
    except Refuse as r:
        return _result(f"REFUSED:{r.code}", r.code, r.detail, **st)
    except Exception as e:  # fail closed on anything unexpected
        return _result("REFUSED:INTERNAL", "INTERNAL", f"{type(e).__name__}", **st)


def _verify(st, guarantee, public_key, cal_rows, monitor_chain, deploy_rows, labels, policy):
    pol = dict(DEFAULT_POLICY)
    if policy is not None:
        if type(policy) is not dict or not set(policy) <= set(DEFAULT_POLICY):
            raise Refuse("SCHEMA", "policy has unknown keys")
        pol.update(policy)
    if guarantee is None or (isinstance(guarantee, (str, bytes)) and not guarantee):
        raise Refuse("MISSING_INPUT", "guarantee envelope not supplied")
    if cal_rows is None:
        raise Refuse("MISSING_INPUT", "calibration rows not supplied")
    if monitor_chain is not None and deploy_rows is None:
        raise Refuse("MISSING_INPUT", "monitor chain supplied without deployment rows")
    if deploy_rows is not None and monitor_chain is None:
        raise Refuse("MISSING_INPUT", "deployment rows supplied without a monitor chain")
    if labels is not None and deploy_rows is None:
        raise Refuse("MISSING_INPUT", "labels supplied without deployment rows")
    pub = _public_key(public_key)

    genv, g = parse_envelope(guarantee, rbcg.PT_GUARANTEE, pub, "guarantee")
    check_guarantee_schema(g)
    gdig = rbcg.envelope_digest(genv)
    st["guarantee_sha256"] = gdig
    p, c, mon = g["procedure"], g["calibration"], g["monitor"]

    # --- what is claimed must be what this verifier supports and the policy requires
    if p["id"] != rbcg.PROCEDURE_ID or p["p_value"] != rbcg.PVALUE_RULE or \
            p["selector"]["id"] != rbcg.SELECTOR_ID:
        raise Refuse("CLAIM_UNSUPPORTED", "unknown calibration procedure, p-value rule or selector")
    grid = p["grid_c"]
    if p["grid_order"] != SUPPORTED_GRID_ORDER or any(a <= b for a, b in zip(grid, grid[1:])):
        raise Refuse("CLAIM_UNSUPPORTED", "grid must be strictly descending (most conservative first)")
    if p["alpha_e6"] != pol["alpha_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", f"claimed alpha_e6={p['alpha_e6']} differs from required "
                                          f"{pol['alpha_e6']}")
    if p["delta_e6"] > pol["delta_max_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", "claimed delta exceeds the policy maximum")
    if mon["score_id"] != rbcg.MONITOR_SCORE_ID or mon["p_value_id"] != rbcg.MONITOR_PVALUE_ID or \
            mon["martingale_id"] != rbcg.MONITOR_MARTINGALE_ID or mon["unit"] != rbcg.MONITOR_UNIT or \
            mon["betting_start"] != "first_deployment_unit":
        raise Refuse("CLAIM_UNSUPPORTED", "unknown monitor specification")
    if mon["delta_s_e6"] > pol["delta_s_max_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", "monitor delta_s exceeds the policy maximum")
    if g["audit"]["spending"] != rbcg.AUDIT_SPENDING:
        raise Refuse("CLAIM_UNSUPPORTED", "unknown audit spending rule")
    if p["tau_star_c"] is None:
        raise Refuse("CLAIM_UNSUPPORTED", "receipt certifies no threshold")
    if g["claim"] != _claim_expected(g):
        raise Refuse("CLAIM_UNSUPPORTED", "claim text does not match the certified parameters")

    # --- bindings: code that produced the receipt, and any expected model hashes
    trusted = pol["trusted_rbcg_sha256"] or [rbcg.rbcg_code_sha256()]
    if g["bindings"]["rbcg_code_sha256"] not in trusted:
        raise Refuse("HASH_MISMATCH", "rbcg.py hash bound in the receipt is not trusted")
    for k, v in (pol["expected_bindings"] or {}).items():
        if g["bindings"].get(k) != v:
            raise Refuse("HASH_MISMATCH", f"binding {k} differs from the expected value")

    # --- calibration rows: schema, hash, counts
    check_rows(cal_rows, rbcg.CAL_ROW_KEYS, "cal_rows")
    if rbcg.sha256_hex(rbcg.canonical_json(cal_rows)) != c["rows_sha256"]:
        raise Refuse("HASH_MISMATCH", "calibration rows do not match the bound rows_sha256")
    if len(cal_rows) != c["n_rows"]:
        raise Refuse("HASH_MISMATCH", "calibration row count differs from the receipt")
    cal_units = _units(cal_rows, "cal_rows")
    if len(cal_units) != c["n_units"] or len(cal_units) != mon["n_cal_units"]:
        raise Refuse("RECOMPUTE_MISMATCH", "calibration unit count differs from the receipt")

    # --- recompute LTT from scratch
    table = recompute_table(cal_rows, grid, p["alpha_e6"], p["selector"]["ood_floor_c"])
    if rbcg.sha256_hex(rbcg.canonical_json(table)) != p["table_sha256"]:
        raise Refuse("RECOMPUTE_MISMATCH", "per-tau (n, k, p) table differs from the receipt")
    tau, n_rej = recompute_fixed_sequence(table, p["delta_e6"])
    if tau != p["tau_star_c"] or n_rej != p["n_rejected"]:
        raise Refuse("RECOMPUTE_MISMATCH", f"recomputed tau*={tau} differs from claimed "
                                           f"{p['tau_star_c']}")
    if mon["theta_seed"] != rbcg.theta_seed_for(c["rows_sha256"]):
        raise Refuse("RECOMPUTE_MISMATCH", "monitor theta_seed is not derived from the calibration rows")
    st["tau_star_c"] = tau
    if monitor_chain is None:
        return "VALID", "guarantee verified (issuance only; no deployment data supplied)"

    # --- deployment rows, labels
    check_rows(deploy_rows, rbcg.DEP_ROW_KEYS, "deploy_rows")
    cal_cids = {r["cid"] for r in cal_rows}
    cal_unit_ids = {u for u, _ in cal_units}
    if any(r["cid"] in cal_cids or r["unit"] in cal_unit_ids for r in deploy_rows):
        raise Refuse("LEAKAGE", "deployment rows overlap the calibration rows")
    dep_units = _units(deploy_rows, "deploy_rows")
    lab = {}
    if labels is not None:
        if type(labels) is not dict:
            raise Refuse("SCHEMA", "labels must be an object {cid: label}")
        dep_cids = {r["cid"] for r in deploy_rows}
        for cid, v in labels.items():
            if cid not in dep_cids:
                raise Refuse("SCHEMA", "label for an unknown deployment cid")
            _str(v, f"labels[{cid}]", LABEL_RE)
        lab = labels
    if type(monitor_chain) is not list or not monitor_chain:
        raise Refuse("SCHEMA", "monitor chain must be a non-empty list of envelopes")

    # --- walk the chain once; record a verdict after every batch
    floor_c, alpha = p["selector"]["ood_floor_c"], p["alpha_e6"] / 1e6
    prev_scores = [_unit_score(rows) for _, rows in cal_units]
    eps = [e / 1000.0 for e in mon["epsilons_e3"]]
    logc = [0.0] * len(eps)
    log_thr = math.log(10 ** 6 / mon["delta_s_e6"])
    prev_digest, alarmed = gdig, False
    n_acc = k_err = 0
    first_event, refusal = None, None
    for j, raw in enumerate(monitor_chain, start=1):
        try:
            if j > len(dep_units):
                raise Refuse("CHAIN_BROKEN", f"monitor receipt {j} has no deployment batch")
            menv, m = parse_envelope(raw, rbcg.PT_MONITOR, pub, f"monitor[{j}]")
            check_monitor_schema(m, f"monitor[{j}]")
            if m["guarantee_sha256"] != gdig:
                raise Refuse("CHAIN_BROKEN", f"monitor[{j}] is bound to a different guarantee")
            if m["seq"] != j:
                raise Refuse("CHAIN_BROKEN", f"monitor[{j}] has seq {m['seq']}")
            if m["prev_sha256"] != prev_digest:
                raise Refuse("CHAIN_BROKEN", f"monitor[{j}] prev_sha256 does not link")
            if pol["require_order_commitment"] and gdig[:16] not in m["deploy_source"]:
                raise Refuse("ORDER", f"monitor[{j}] deployment source does not commit to the "
                                      f"guarantee digest (data not provably drawn after issuance)")
            unit_id, rows = dep_units[j - 1]
            if m["unit_id"] != unit_id or m["batch_n_rows"] != len(rows) or \
                    m["batch_sha256"] != rbcg.sha256_hex(rbcg.canonical_json(rows)):
                raise Refuse("HASH_MISMATCH", f"monitor[{j}] does not bind deployment batch {j}")
            s = _unit_score(rows)
            pj = _p_direct(prev_scores, s, _theta(mon["theta_seed"], j))
            lp = math.log(pj)
            for i, e in enumerate(eps):
                logc[i] += math.log(e) + (e - 1.0) * lp
            mx = max(logc)
            log_m = mx + math.log(math.fsum(math.exp(v - mx) for v in logc)) - math.log(len(eps))
            alarmed = alarmed or log_m >= log_thr
            if s != m["score"] or abs(int(round(pj * rbcg.E12)) - m["p_e12"]) > 2 or \
                    abs(int(round(log_m * 1e9)) - m["log_m_e9"]) > 2 or alarmed != m["alarm"]:
                raise Refuse("RECOMPUTE_MISMATCH", f"monitor[{j}] score/p/martingale/alarm differ "
                                                   f"from recomputation")
        except Refuse as r:
            refusal = r
            st["trace"].append({"batch": j, "verdict": f"REFUSED:{r.code}"})
            break
        prev_scores.append(s)
        prev_digest = rbcg.envelope_digest(menv)
        st["n_batches"] = j
        if alarmed and st["alarm_batch"] is None:
            st["alarm_batch"] = j
            first_event = first_event or ("REFUSED:SHIFT_ALARM", j)
        # label audit (labels of batch j arrive after its unlabelled data were monitored)
        for r in rows:
            if r["cid"] in lab and rbcg.accepts(r, tau, floor_c):
                n_acc += 1
                k_err += r["pred"] != lab[r["cid"]]
        d_j = pol["delta_audit_e6"] / 1e6 * 6.0 / (math.pi ** 2 * j * j)
        if n_acc > 0 and st["violation_batch"] is None and rbcg.binom_sf(k_err, n_acc, alpha) < d_j:
            st["violation_batch"] = j
            first_event = first_event or ("VIOLATED", j)
        st["audit"] = {"n_accepted_labelled": n_acc, "errors": k_err, "look": j,
                       "delta_j_e9": int(d_j * 1e9)}
        st["trace"].append({"batch": j, "verdict": first_event[0] if first_event else "VALID",
                            "log_m_e9": m["log_m_e9"], "n_acc": n_acc, "k_err": k_err})
    if refusal is not None:
        raise refusal
    if len(monitor_chain) != len(dep_units):
        st["trace"].append({"batch": len(monitor_chain) + 1, "verdict": "REFUSED:CHAIN_BROKEN"})
        raise Refuse("CHAIN_BROKEN", f"{len(dep_units)} deployment batches but "
                                     f"{len(monitor_chain)} monitor receipts")
    if st["audit"] is not None and st["audit"]["n_accepted_labelled"] > 0:
        a = st["audit"]
        st["audit"]["cp_lower_e9"] = int(rbcg.cp_lower(a["errors"], a["n_accepted_labelled"],
                                                       a["delta_j_e9"] / 1e9) * 1e9)
    if first_event is None:
        return "VALID", f"guarantee and {st['n_batches']} monitor receipts verified; no alarm, no violation"
    if first_event[0] == "REFUSED:SHIFT_ALARM":
        raise Refuse("SHIFT_ALARM", f"exchangeability monitor alarmed at batch {first_event[1]}")
    return "VIOLATED", f"label audit: CP lower bound above alpha at batch {first_event[1]}"


# ------------------------------------------------------------------ CLI
def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _read_rows(path):
    out = []
    for i, line in enumerate(_read(path).decode("utf-8").splitlines()):
        if line.strip():
            out.append(loads_strict(line, f"{os.path.basename(path)}:{i + 1}"))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Fail-closed RBCG v0 verifier (synthetic data only)")
    ap.add_argument("--guarantee", required=True)
    ap.add_argument("--pubkey", required=True, help="PEM file or file holding SEC1 hex")
    ap.add_argument("--cal-rows", required=True, help="JSONL, one row per line")
    ap.add_argument("--monitor-chain", help="JSONL, one envelope per line")
    ap.add_argument("--deploy-rows", help="JSONL")
    ap.add_argument("--labels", help="JSON object {cid: label}")
    ap.add_argument("--policy", help="JSON object overriding DEFAULT_POLICY keys")
    a = ap.parse_args(argv)
    try:
        pk = _read(a.pubkey).decode("ascii").strip()
        if pk.startswith("-----BEGIN"):
            pk += "\n"
        res = verify(_read(a.guarantee), pk, _read_rows(a.cal_rows),
                     [ln for ln in _read(a.monitor_chain).splitlines() if ln.strip()]
                     if a.monitor_chain else None,
                     _read_rows(a.deploy_rows) if a.deploy_rows else None,
                     loads_strict(_read(a.labels), "labels") if a.labels else None,
                     loads_strict(_read(a.policy), "policy") if a.policy else None)
    except Refuse as r:
        res = _result(f"REFUSED:{r.code}", r.code, r.detail)
    except Exception as e:
        res = _result("REFUSED:MISSING_INPUT" if isinstance(e, OSError) else "REFUSED:INTERNAL",
                      "MISSING_INPUT" if isinstance(e, OSError) else "INTERNAL", type(e).__name__)
    res = dict(res)
    res["trace"] = res["trace"][-5:]
    print(json.dumps(res, sort_keys=True))
    return 0 if res["verdict"] == "VALID" else (1 if res["verdict"] == "VIOLATED" else 2)


if __name__ == "__main__":
    sys.exit(main())
