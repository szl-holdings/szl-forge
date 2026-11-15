#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1_verify.py -- fail-closed offline verifier for RBCG v1 guarantee, monitor and log receipts.

IN-SILICO RESEARCH ONLY. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE.

Inputs: guarantee envelope, public key, calibration rows; optionally a monitor chain, deployment rows,
labels, and a transparency-log anchor (two signed tree heads, inclusion proofs for the guarantee and
the monitor-chain head, and a consistency proof) with the log public key. Verdicts:
  VALID                 every check passed and no event fired
  VIOLATED              labels arrived and, while the guarantee was in force, the anytime-valid betting
                        confidence sequence for the scene-level linearized selective loss E[L] at tau*
                        excluded 0 from below (K_t(alpha) >= 1/delta_audit)
  REFUSED:<CODE>        MISSING_INPUT, SCHEMA, BAD_SIGNATURE, HASH_MISMATCH, RECOMPUTE_MISMATCH,
                        CLAIM_UNSUPPORTED, CHAIN_BROKEN, ORDER, LEAKAGE, NOT_INCLUDED, INCONSISTENT_STH,
                        SHIFT_ALARM, INTERNAL
Every exception becomes a refusal; nothing raises out of verify(). Floats, NaN, Infinity, duplicate
keys, missing or extra fields and wrong types are refused in every document. Per-scene counts, scores,
conformal p-values and the label-audit losses are recomputed by direct definitions written
independently of rbcg1.py's fast paths; the floating-point kernels (betting capital, HB, CS) are
shared and are cross-checked against independent references in the test suite.
"""
import base64
import hashlib
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbcg1  # noqa: E402
import rbcg1_log  # noqa: E402

CODES = ("MISSING_INPUT", "SCHEMA", "BAD_SIGNATURE", "HASH_MISMATCH", "RECOMPUTE_MISMATCH",
         "CLAIM_UNSUPPORTED", "CHAIN_BROKEN", "ORDER", "LEAKAGE", "NOT_INCLUDED", "INCONSISTENT_STH",
         "SHIFT_ALARM", "INTERNAL")
DEFAULT_POLICY = {"alpha_e6": 100000, "delta_max_e6": 100000, "delta_s_max_e6": 10000,
                  "delta_audit_e6": 50000, "audit_c_e3": 750, "require_order_commitment": True,
                  "require_anchor": True, "trusted_rbcg_sha256": None, "expected_bindings": {},
                  "expected_protocol_sha256": None, "unit_rows_max": 24}
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
    return check_envelope_obj(loads_strict(raw, what), expected_type, pub, what)


def check_envelope_obj(env, expected_type, pub, what):
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
    if not pub.verify(rbcg1.pae(expected_type, body), sig):
        raise Refuse("BAD_SIGNATURE", f"{what}: ECDSA P-256 signature over the DSSE PAE is invalid")
    payload = loads_strict(body, f"{what}.payload")
    if rbcg1.canonical_json(payload) != body:
        raise Refuse("SCHEMA", f"{what}: payload is not canonical JSON")
    return env, payload


def _public_key(pk):
    if pk is None:
        raise Refuse("MISSING_INPUT", "public key not supplied")
    try:
        if isinstance(pk, rbcg1.PublicKey):
            return pk
        if type(pk) is str and pk.lstrip().startswith("-----BEGIN"):
            return rbcg1.PublicKey.from_pem(pk)
        if type(pk) is str:
            return rbcg1.PublicKey.from_sec1_hex(pk.strip())
    except ValueError as e:
        raise Refuse("BAD_SIGNATURE", f"public key invalid: {e}")
    raise Refuse("SCHEMA", "public key must be a PEM or SEC1 hex string")


# ------------------------------------------------------------------ payload schemas
BINDING_KEYS = ("advisory_code_sha256", "governance_code_sha256", "model_id", "model_sha256",
                "rbcg_code_sha256", "sim_code_sha256", "weights_file_sha256")
G_KEYS = ("anchoring", "audit", "bindings", "calibration", "claim", "issued_at_utc", "monitor",
          "not_for_clinical_use", "procedure", "protocol_sha256", "rbcg_version", "schema", "synthetic_only")
CAL_KEYS = ("env_profile", "n_rows", "n_units", "row_schema", "rows_sha256", "seed_family", "unit_rows_max")
PROC_KEYS = ("alpha_e6", "bet_c_e3", "delta_e6", "grid_c", "grid_order", "id", "loss", "n_rejected",
             "p_value", "secondary_n_rejected", "secondary_p_value", "secondary_tau_star_c", "selector",
             "table_sha256", "tau_star_c")
MON_KEYS = ("betting_start", "combination", "epsilons_e3", "martingale_id", "n_cal_units", "p_value_id",
            "primary", "secondary", "unit")
COMP_SPEC_KEYS = ("delta_s_e6", "score_id", "theta_seed")
AUDIT_KEYS = ("bet_c_e3", "delta_audit_e6", "id", "loss", "unit")
ANCHORING_KEYS = ("leaf_rule", "log_id")
M_KEYS = ("alarm", "batch_n_rows", "batch_sha256", "deploy_source", "guarantee_sha256", "prev_sha256",
          "primary", "schema", "secondary", "seq", "unit_id")
M_COMP_KEYS = ("alarm", "log_m_e9", "p_e12", "score")
STH_KEYS = ("log_id", "root_hash", "schema", "timestamp_utc", "tree_size")
ANCHOR_DOC_KEYS = ("consistency", "guarantee_inclusion", "head_inclusion", "schema", "sth_final", "sth_issue")
INCL_KEYS = ("leaf_index", "path", "tree_size")
CONS_KEYS = ("first_size", "path", "second_size")
COMPONENTS = ("primary", "secondary")


def _opt_tau(v, path):
    if v is not None:
        _int(v, path, 1, 100)


def check_guarantee_schema(g):
    _obj(g, G_KEYS, "guarantee")
    if g["schema"] != rbcg1.GUARANTEE_SCHEMA:
        raise Refuse("SCHEMA", "guarantee.schema")
    _str(g["rbcg_version"], "rbcg_version")
    _bool(g["synthetic_only"], "synthetic_only")
    _bool(g["not_for_clinical_use"], "not_for_clinical_use")
    if g["synthetic_only"] is not True or g["not_for_clinical_use"] is not True:
        raise Refuse("CLAIM_UNSUPPORTED", "receipt does not declare synthetic-only, non-clinical use")
    _str(g["claim"], "claim")
    if type(g["issued_at_utc"]) is not str:
        raise Refuse("SCHEMA", "issued_at_utc must be a string")
    _str(g["protocol_sha256"], "protocol_sha256", HEX64)
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
    _int(c["unit_rows_max"], "calibration.unit_rows_max", 1, 100000)
    _str(c["rows_sha256"], "calibration.rows_sha256", HEX64)
    if c["row_schema"] != rbcg1.ROW_SCHEMA:
        raise Refuse("SCHEMA", "calibration.row_schema")
    p = g["procedure"]
    _obj(p, PROC_KEYS, "procedure")
    _int(p["alpha_e6"], "procedure.alpha_e6", 1, 999999)
    _int(p["delta_e6"], "procedure.delta_e6", 1, 999999)
    _int(p["bet_c_e3"], "procedure.bet_c_e3", 1, 999)
    if type(p["grid_c"]) is not list or not p["grid_c"]:
        raise Refuse("SCHEMA", "procedure.grid_c must be a non-empty list")
    for i, t in enumerate(p["grid_c"]):
        _int(t, f"procedure.grid_c[{i}]", 1, 100)
    for k in ("grid_order", "id", "loss", "p_value", "secondary_p_value"):
        _str(p[k], f"procedure.{k}")
    _obj(p["selector"], ("id", "ood_floor_c"), "procedure.selector")
    _str(p["selector"]["id"], "procedure.selector.id")
    _int(p["selector"]["ood_floor_c"], "procedure.selector.ood_floor_c", 0, 100)
    _str(p["table_sha256"], "procedure.table_sha256", HEX64)
    _int(p["n_rejected"], "procedure.n_rejected", 0)
    _int(p["secondary_n_rejected"], "procedure.secondary_n_rejected", 0)
    _opt_tau(p["tau_star_c"], "procedure.tau_star_c")
    _opt_tau(p["secondary_tau_star_c"], "procedure.secondary_tau_star_c")
    m = g["monitor"]
    _obj(m, MON_KEYS, "monitor")
    for k in ("betting_start", "combination", "martingale_id", "p_value_id", "unit"):
        _str(m[k], f"monitor.{k}")
    if type(m["epsilons_e3"]) is not list or not m["epsilons_e3"]:
        raise Refuse("SCHEMA", "monitor.epsilons_e3 must be a non-empty list")
    for i, e in enumerate(m["epsilons_e3"]):
        _int(e, f"monitor.epsilons_e3[{i}]", 1, 999)
    _int(m["n_cal_units"], "monitor.n_cal_units", 1)
    for name in COMPONENTS:
        cs = m[name]
        _obj(cs, COMP_SPEC_KEYS, f"monitor.{name}")
        _str(cs["score_id"], f"monitor.{name}.score_id")
        _int(cs["delta_s_e6"], f"monitor.{name}.delta_s_e6", 1, 999999)
        _str(cs["theta_seed"], f"monitor.{name}.theta_seed", HEX64)
    a = g["audit"]
    _obj(a, AUDIT_KEYS, "audit")
    for k in ("id", "loss", "unit"):
        _str(a[k], f"audit.{k}")
    _int(a["delta_audit_e6"], "audit.delta_audit_e6", 1, 999999)
    _int(a["bet_c_e3"], "audit.bet_c_e3", 1, 999)
    an = g["anchoring"]
    _obj(an, ANCHORING_KEYS, "anchoring")
    _str(an["log_id"], "anchoring.log_id", HEX64)
    _str(an["leaf_rule"], "anchoring.leaf_rule")


def check_monitor_schema(m, path):
    _obj(m, M_KEYS, path)
    if m["schema"] != rbcg1.MONITOR_SCHEMA:
        raise Refuse("SCHEMA", f"{path}.schema")
    _bool(m["alarm"], f"{path}.alarm")
    _int(m["batch_n_rows"], f"{path}.batch_n_rows", 1)
    for k in ("batch_sha256", "guarantee_sha256", "prev_sha256"):
        _str(m[k], f"{path}.{k}", HEX64)
    _str(m["deploy_source"], f"{path}.deploy_source", ID_RE)
    _str(m["unit_id"], f"{path}.unit_id", ID_RE)
    _int(m["seq"], f"{path}.seq", 1)
    for name in COMPONENTS:
        c = m[name]
        _obj(c, M_COMP_KEYS, f"{path}.{name}")
        _bool(c["alarm"], f"{path}.{name}.alarm")
        _int(c["score"], f"{path}.{name}.score", 0, rbcg1.E12)
        _int(c["p_e12"], f"{path}.{name}.p_e12", 0, rbcg1.E12)
        _int(c["log_m_e9"], f"{path}.{name}.log_m_e9")


def check_sth_schema(s, path):
    _obj(s, STH_KEYS, path)
    if s["schema"] != rbcg1_log.STH_SCHEMA:
        raise Refuse("SCHEMA", f"{path}.schema")
    _str(s["log_id"], f"{path}.log_id", HEX64)
    _int(s["tree_size"], f"{path}.tree_size", 0)
    _str(s["root_hash"], f"{path}.root_hash", HEX64)
    if type(s["timestamp_utc"]) is not str:
        raise Refuse("SCHEMA", f"{path}.timestamp_utc must be a string")


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
        _int(r["pmax_e12"], f"{path}.pmax_e12", 0, rbcg1.E12)
        _str(r["pred"], f"{path}.pred", LABEL_RE)
        if "label" in keys:
            _str(r["label"], f"{path}.label", LABEL_RE)
        if r["cid"] in seen:
            raise Refuse("SCHEMA", f"{path}.cid duplicated")
        seen.add(r["cid"])


def _units(rows, what):
    try:
        return rbcg1.group_units(rows)
    except ValueError as e:
        raise Refuse("SCHEMA", f"{what}: {e}")


def _units(rows, what):
    try:
        return rbcg1.group_units(rows)
    except ValueError as e:
        raise Refuse("SCHEMA", f"{what}: {e}")


# ------------------------------------------------------------------ independent recomputation
def recompute_unit_counts(units, grid, floor_c):
    """Direct per-unit (accepted, errors) for each tau of a strictly descending grid by sort-and-sweep
    over exact integer comparisons pmax_e12 >= max(tau, floor if ood) * 1e10 (independent of the
    issuer's histogram fast path)."""
    res = [[] for _ in grid]
    for _, rows in units:
        cand = sorted(((r["pmax_e12"], int(r["pred"] != r["label"])) for r in rows
                       if r["inf"] and (not r["ood"] or r["pmax_e12"] >= floor_c * 10 ** 10)), reverse=True)
        i = a = e = 0
        for ti, t in enumerate(grid):
            thr = t * 10 ** 10
            while i < len(cand) and cand[i][0] >= thr:
                a += 1
                e += cand[i][1]
                i += 1
            res[ti].append((a, e))
    return res


def recompute_table(units, grid, alpha_e6, delta_e6, floor_c, n_max, c_e3):
    alpha, delta, c = alpha_e6 / 1e6, delta_e6 / 1e6, c_e3 / 1000.0
    den = n_max * 10 ** 6
    out = []
    for t, cs in zip(grid, recompute_unit_counts(units, grid, floor_c)):
        nums = [e * 10 ** 6 - alpha_e6 * a + alpha_e6 * n_max for a, e in cs]
        logk = rbcg1.bet_log_kmax([v / den for v in nums], alpha, delta, c)
        out.append({"k_err": sum(e for _, e in cs), "logk_e9": int(math.floor(logk * 1e9)),
                    "n_acc": sum(a for a, _ in cs), "n_units": len(cs),
                    "p_bet_e9": int(math.ceil(min(1.0, math.exp(-logk)) * 1e9)),
                    "p_hb_e9": int(math.ceil(rbcg1.hb_pvalue(nums, den, alpha_e6) * 1e9)),
                    "sum_x_num": sum(nums), "tau_c": t})
    return out


def recompute_fixed_sequence(table, delta_e6, key):
    tau, n_rej = None, 0
    for e in table:
        if e[key] > delta_e6 * 1000:
            break
        tau, n_rej = e["tau_c"], n_rej + 1
    return tau, n_rej


def _theta(seed, j):
    d = hashlib.sha256(f"{seed}|{j}".encode("ascii")).digest()
    return ((int.from_bytes(d[:7], "big") >> 3) + 1) / 2.0 ** 53


def _theta_seed(domain, rows_sha):
    return hashlib.sha256((domain + "|" + rows_sha).encode("ascii")).hexdigest()


def _p_direct(prev_scores, s, th):
    gt = sum(1 for x in prev_scores if x > s)
    eq = sum(1 for x in prev_scores if x == s) + 1
    return (gt + th * eq) / (len(prev_scores) + 1)


def _score(name, rows):
    if name == "primary":
        return sum(1 for r in rows if r["ood"] is True)
    return sum(rbcg1.E12 - r["pmax_e12"] for r in rows) // len(rows)


def _scene_x(rows, lab, tau, floor_c, alpha_e6, n_max):
    """Direct scene-level linearized loss X = (e - alpha a)/n_max + alpha at tau, or None when an
    accepted row of the scene has no label (the scene is then not audited)."""
    a = e = 0
    for r in rows:
        if r["inf"] and r["pmax_e12"] >= (max(tau, floor_c) if r["ood"] else tau) * 10 ** 10:
            if r["cid"] not in lab:
                return None
            a += 1
            e += r["pred"] != lab[r["cid"]]
    return (e * 10 ** 6 - alpha_e6 * a + alpha_e6 * n_max) / (n_max * 10 ** 6)


def _claim_expected(g):
    p, c, m = g["procedure"], g["calibration"], g["monitor"]
    return rbcg1.claim_text(g["bindings"]["model_id"], p["tau_star_c"], p["selector"]["ood_floor_c"],
                            p["alpha_e6"], p["delta_e6"], c["n_units"], c["n_rows"], c["seed_family"],
                            m["primary"]["delta_s_e6"], m["secondary"]["delta_s_e6"])


# ------------------------------------------------------------------ transparency-log anchor
def _hexes(v, path):
    if type(v) is not list or len(v) > 64:
        raise Refuse("SCHEMA", f"{path} must be a list of at most 64 hashes")
    for i, h in enumerate(v):
        _str(h, f"{path}[{i}]", HEX64)
    return [bytes.fromhex(h) for h in v]


def _sth(env_obj, log_pub, what):
    _, s = check_envelope_obj(env_obj, rbcg1.PT_STH, log_pub, what)
    check_sth_schema(s, what)
    return s


def _incl(d, what):
    _obj(d, INCL_KEYS, what)
    _int(d["leaf_index"], f"{what}.leaf_index", 0)
    _int(d["tree_size"], f"{what}.tree_size", 0)
    return _hexes(d["path"], f"{what}.path")


def check_anchor(anchor, log_pub, g, genv, head_env):
    """Inclusion of the guarantee in the issuance STH, of the monitor-chain head in the final STH,
    and consistency issuance STH -> final STH (RFC 9162 2.1.3.2 / 2.1.4.2)."""
    if isinstance(anchor, (str, bytes, bytearray)):
        anchor = loads_strict(anchor, "anchor")
    _obj(anchor, ANCHOR_DOC_KEYS, "anchor")
    if anchor["schema"] != rbcg1_log.ANCHOR_SCHEMA:
        raise Refuse("SCHEMA", "anchor.schema")
    s1 = _sth(anchor["sth_issue"], log_pub, "sth_issue")
    gpath = _incl(anchor["guarantee_inclusion"], "anchor.guarantee_inclusion")
    gi = anchor["guarantee_inclusion"]
    if s1["log_id"] != log_pub.keyid() or s1["log_id"] != g["anchoring"]["log_id"]:
        raise Refuse("NOT_INCLUDED", "the issuance STH is not from the log named by the guarantee")
    if gi["tree_size"] != s1["tree_size"]:
        raise Refuse("INCONSISTENT_STH", "guarantee inclusion proof is not for the issuance STH size")
    g_leaf = rbcg1_log.leaf_hash(rbcg1.envelope_bytes(genv))
    if not rbcg1_log.verify_inclusion(gi["leaf_index"], gi["tree_size"], g_leaf, gpath,
                                      bytes.fromhex(s1["root_hash"])):
        raise Refuse("NOT_INCLUDED", "guarantee receipt is not included in the issuance STH")
    out = {"log_id": s1["log_id"], "sth_issue_size": s1["tree_size"], "guarantee_index": gi["leaf_index"],
           "sth_final_size": None, "head_index": None}
    if head_env is None:
        if any(anchor[k] is not None for k in ("sth_final", "head_inclusion", "consistency")):
            raise Refuse("SCHEMA", "an anchor without a monitor chain must not carry final STH or proofs")
        return out
    if any(anchor[k] is None for k in ("sth_final", "head_inclusion", "consistency")):
        raise Refuse("MISSING_INPUT", "anchor lacks the final STH, head inclusion or consistency proof")
    s2 = _sth(anchor["sth_final"], log_pub, "sth_final")
    hpath = _incl(anchor["head_inclusion"], "anchor.head_inclusion")
    co = anchor["consistency"]
    _obj(co, CONS_KEYS, "anchor.consistency")
    _int(co["first_size"], "anchor.consistency.first_size", 0)
    _int(co["second_size"], "anchor.consistency.second_size", 0)
    cpath = _hexes(co["path"], "anchor.consistency.path")
    hi = anchor["head_inclusion"]
    if s2["log_id"] != s1["log_id"]:
        raise Refuse("INCONSISTENT_STH", "the two STHs come from different logs")
    if s2["tree_size"] < s1["tree_size"] or co["first_size"] != s1["tree_size"] or \
            co["second_size"] != s2["tree_size"] or hi["tree_size"] != s2["tree_size"]:
        raise Refuse("INCONSISTENT_STH", "proof sizes do not match the signed tree heads")
    if not rbcg1_log.verify_consistency(s1["tree_size"], s2["tree_size"], bytes.fromhex(s1["root_hash"]),
                                        bytes.fromhex(s2["root_hash"]), cpath):
        raise Refuse("INCONSISTENT_STH", "final STH is not an append-only extension of the issuance STH")
    h_leaf = rbcg1_log.leaf_hash(rbcg1.envelope_bytes(head_env))
    if not rbcg1_log.verify_inclusion(hi["leaf_index"], hi["tree_size"], h_leaf, hpath,
                                      bytes.fromhex(s2["root_hash"])):
        raise Refuse("NOT_INCLUDED", "monitor-chain head is not included in the final STH")
    if hi["leaf_index"] < s1["tree_size"]:
        raise Refuse("ORDER", "monitor-chain head was logged before the issuance STH")
    out.update({"sth_final_size": s2["tree_size"], "head_index": hi["leaf_index"]})
    return out


# ------------------------------------------------------------------ main entry
def _result(verdict, code=None, detail="", **kw):
    out = {"verdict": verdict, "code": code, "detail": detail, "guarantee_sha256": None,
           "tau_star_c": None, "n_batches": 0, "alarm_batch": None, "alarm_component": None,
           "violation_batch": None, "trace": [], "audit": None, "anchor": None}
    out.update(kw)
    return out


def verify(guarantee, public_key, cal_rows, monitor_chain=None, deploy_rows=None, labels=None,
           policy=None, anchor=None, log_public_key=None):
    """Never raises. Returns a dict with 'verdict' in {VALID, VIOLATED, REFUSED:<CODE>}."""
    st = {"guarantee_sha256": None, "tau_star_c": None, "n_batches": 0, "alarm_batch": None,
          "alarm_component": None, "violation_batch": None, "trace": [], "audit": None, "anchor": None}
    try:
        verdict, detail = _verify(st, guarantee, public_key, cal_rows, monitor_chain, deploy_rows,
                                  labels, policy, anchor, log_public_key)
        return _result(verdict, None, detail, **st)
    except Refuse as r:
        return _result(f"REFUSED:{r.code}", r.code, r.detail, **st)
    except Exception as e:  # fail closed on anything unexpected
        return _result("REFUSED:INTERNAL", "INTERNAL", f"{type(e).__name__}", **st)


def _verify(st, guarantee, public_key, cal_rows, monitor_chain, deploy_rows, labels, policy, anchor,
            log_public_key):
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
    if monitor_chain is not None and pol["require_anchor"] and anchor is None:
        raise Refuse("MISSING_INPUT", "transparency-log anchor required by policy but not supplied")
    if anchor is not None and log_public_key is None:
        raise Refuse("MISSING_INPUT", "anchor supplied without the log public key")
    pub = _public_key(public_key)
    log_pub = _public_key(log_public_key) if anchor is not None else None

    genv, g = parse_envelope(guarantee, rbcg1.PT_GUARANTEE, pub, "guarantee")
    check_guarantee_schema(g)
    gdig = rbcg1.envelope_digest(genv)
    st["guarantee_sha256"] = gdig
    p, c, mon, aud = g["procedure"], g["calibration"], g["monitor"], g["audit"]

    # --- what is claimed must be what this verifier supports and the policy requires
    if p["id"] != rbcg1.PROCEDURE_ID or p["loss"] != rbcg1.LOSS_ID or p["p_value"] != rbcg1.PVALUE_RULE or \
            p["secondary_p_value"] != rbcg1.SECONDARY_PVALUE_RULE or p["selector"]["id"] != rbcg1.SELECTOR_ID:
        raise Refuse("CLAIM_UNSUPPORTED", "unknown calibration procedure, loss, p-value rule or selector")
    grid = p["grid_c"]
    if p["grid_order"] != SUPPORTED_GRID_ORDER or any(a <= b for a, b in zip(grid, grid[1:])):
        raise Refuse("CLAIM_UNSUPPORTED", "grid must be strictly descending (most conservative first)")
    if p["alpha_e6"] != pol["alpha_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", f"claimed alpha_e6={p['alpha_e6']} differs from required "
                                          f"{pol['alpha_e6']}")
    if p["delta_e6"] > pol["delta_max_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", "claimed delta exceeds the policy maximum")
    if c["unit_rows_max"] != pol["unit_rows_max"]:
        raise Refuse("CLAIM_UNSUPPORTED", "unit_rows_max differs from the policy")
    if mon["primary"]["score_id"] != rbcg1.MONITOR_PRIMARY_SCORE_ID or \
            mon["secondary"]["score_id"] != rbcg1.MONITOR_SECONDARY_SCORE_ID or \
            mon["p_value_id"] != rbcg1.MONITOR_PVALUE_ID or mon["martingale_id"] != rbcg1.MONITOR_MARTINGALE_ID or \
            mon["combination"] != rbcg1.MONITOR_COMBINATION_ID or mon["unit"] != rbcg1.MONITOR_UNIT or \
            mon["betting_start"] != "first_deployment_unit":
        raise Refuse("CLAIM_UNSUPPORTED", "unknown monitor specification")
    if mon["primary"]["delta_s_e6"] + mon["secondary"]["delta_s_e6"] > pol["delta_s_max_e6"]:
        raise Refuse("CLAIM_UNSUPPORTED", "monitor delta_s (primary + secondary) exceeds the policy maximum")
    if aud["id"] != rbcg1.AUDIT_ID or aud["loss"] != rbcg1.LOSS_ID or aud["unit"] != rbcg1.MONITOR_UNIT:
        raise Refuse("CLAIM_UNSUPPORTED", "unknown audit rule")
    if g["anchoring"]["leaf_rule"] != rbcg1.ANCHOR_LEAF_RULE:
        raise Refuse("CLAIM_UNSUPPORTED", "unknown transparency-log leaf rule")
    if p["tau_star_c"] is None:
        raise Refuse("CLAIM_UNSUPPORTED", "receipt certifies no threshold")
    if g["claim"] != _claim_expected(g):
        raise Refuse("CLAIM_UNSUPPORTED", "claim text does not match the certified parameters")

    # --- bindings
    trusted = pol["trusted_rbcg_sha256"] or [rbcg1.rbcg_code_sha256()]
    if g["bindings"]["rbcg_code_sha256"] not in trusted:
        raise Refuse("HASH_MISMATCH", "rbcg1.py hash bound in the receipt is not trusted")
    for k, v in (pol["expected_bindings"] or {}).items():
        if g["bindings"].get(k) != v:
            raise Refuse("HASH_MISMATCH", f"binding {k} differs from the expected value")
    if pol["expected_protocol_sha256"] is not None and g["protocol_sha256"] != pol["expected_protocol_sha256"]:
        raise Refuse("HASH_MISMATCH", "guarantee is not bound to the expected pre-registered protocol")

    # --- calibration rows: schema, hash, counts, unit sizes
    check_rows(cal_rows, rbcg1.CAL_ROW_KEYS, "cal_rows")
    if rbcg1.sha256_hex(rbcg1.canonical_json(cal_rows)) != c["rows_sha256"]:
        raise Refuse("HASH_MISMATCH", "calibration rows do not match the bound rows_sha256")
    if len(cal_rows) != c["n_rows"]:
        raise Refuse("HASH_MISMATCH", "calibration row count differs from the receipt")
    cal_units = _units(cal_rows, "cal_rows")
    n_max = c["unit_rows_max"]
    if any(len(u) > n_max for _, u in cal_units):
        raise Refuse("SCHEMA", "a calibration unit exceeds unit_rows_max (loss bound would not hold)")
    if len(cal_units) != c["n_units"] or len(cal_units) != mon["n_cal_units"]:
        raise Refuse("RECOMPUTE_MISMATCH", "calibration unit count differs from the receipt")

    # --- recompute scene-level LTT from scratch
    floor_c = p["selector"]["ood_floor_c"]
    table = recompute_table(cal_units, grid, p["alpha_e6"], p["delta_e6"], floor_c, n_max, p["bet_c_e3"])
    if rbcg1.sha256_hex(rbcg1.canonical_json(table)) != p["table_sha256"]:
        raise Refuse("RECOMPUTE_MISMATCH", "per-tau scene-level table differs from the receipt")
    tau, n_rej = recompute_fixed_sequence(table, p["delta_e6"], "p_bet_e9")
    if tau != p["tau_star_c"] or n_rej != p["n_rejected"]:
        raise Refuse("RECOMPUTE_MISMATCH", f"recomputed tau*={tau} differs from claimed {p['tau_star_c']}")
    tau2, n_rej2 = recompute_fixed_sequence(table, p["delta_e6"], "p_hb_e9")
    if tau2 != p["secondary_tau_star_c"] or n_rej2 != p["secondary_n_rejected"]:
        raise Refuse("RECOMPUTE_MISMATCH", "recomputed secondary (HB) tau* differs from the receipt")
    if mon["primary"]["theta_seed"] != _theta_seed(rbcg1.THETA_DOMAIN_PRIMARY, c["rows_sha256"]) or \
            mon["secondary"]["theta_seed"] != _theta_seed(rbcg1.THETA_DOMAIN_SECONDARY, c["rows_sha256"]):
        raise Refuse("RECOMPUTE_MISMATCH", "monitor theta seeds are not derived from the calibration rows")
    st["tau_star_c"] = tau
    if monitor_chain is None:
        if anchor is not None:
            st["anchor"] = check_anchor(anchor, log_pub, g, genv, None)
        return "VALID", "guarantee verified (issuance only; no deployment data supplied)"

    # --- deployment rows, labels
    check_rows(deploy_rows, rbcg1.DEP_ROW_KEYS, "deploy_rows")
    cal_cids = {r["cid"] for r in cal_rows}
    cal_unit_ids = {u for u, _ in cal_units}
    if any(r["cid"] in cal_cids or r["unit"] in cal_unit_ids for r in deploy_rows):
        raise Refuse("LEAKAGE", "deployment rows overlap the calibration rows")
    dep_units = _units(deploy_rows, "deploy_rows")
    if any(len(u) > n_max for _, u in dep_units):
        raise Refuse("SCHEMA", "a deployment unit exceeds unit_rows_max")
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
    alpha_e6 = p["alpha_e6"]
    prev = {name: [_score(name, rows) for _, rows in cal_units] for name in COMPONENTS}
    eps = [e / 1000.0 for e in mon["epsilons_e3"]]
    logc = {name: [0.0] * len(eps) for name in COMPONENTS}
    thr = {name: math.log(10 ** 6 / mon[name]["delta_s_e6"]) for name in COMPONENTS}
    comp_alarm = {name: False for name in COMPONENTS}
    audit = rbcg1.BettingAudit(alpha_e6, pol["delta_audit_e6"], pol["audit_c_e3"])
    prev_digest, alarmed, head_env = gdig, False, None
    first_event, refusal = None, None
    for j, raw in enumerate(monitor_chain, start=1):
        try:
            if j > len(dep_units):
                raise Refuse("CHAIN_BROKEN", f"monitor receipt {j} has no deployment batch")
            menv, m = parse_envelope(raw, rbcg1.PT_MONITOR, pub, f"monitor[{j}]")
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
                    m["batch_sha256"] != rbcg1.sha256_hex(rbcg1.canonical_json(rows)):
                raise Refuse("HASH_MISMATCH", f"monitor[{j}] does not bind deployment batch {j}")
            got = {}
            for name in COMPONENTS:
                s = _score(name, rows)
                pj = _p_direct(prev[name], s, _theta(mon[name]["theta_seed"], j))
                lp = math.log(pj)
                for i, e in enumerate(eps):
                    logc[name][i] += math.log(e) + (e - 1.0) * lp
                mx = max(logc[name])
                log_m = mx + math.log(math.fsum(math.exp(v - mx) for v in logc[name])) - math.log(len(eps))
                comp_alarm[name] = comp_alarm[name] or log_m >= thr[name]
                rc = m[name]
                if s != rc["score"] or abs(int(round(pj * rbcg1.E12)) - rc["p_e12"]) > 2 or \
                        abs(int(round(log_m * 1e9)) - rc["log_m_e9"]) > 2 or comp_alarm[name] != rc["alarm"]:
                    raise Refuse("RECOMPUTE_MISMATCH", f"monitor[{j}].{name} score/p/martingale/alarm differ "
                                                       f"from recomputation")
                got[name] = s
            alarmed = alarmed or comp_alarm["primary"] or comp_alarm["secondary"]
            if alarmed != m["alarm"]:
                raise Refuse("RECOMPUTE_MISMATCH", f"monitor[{j}] combined alarm differs from recomputation")
        except Refuse as r:
            refusal = r
            st["trace"].append({"batch": j, "verdict": f"REFUSED:{r.code}"})
            break
        for name in COMPONENTS:
            prev[name].append(got[name])
        prev_digest = rbcg1.envelope_digest(menv)
        head_env = menv
        st["n_batches"] = j
        if alarmed and st["alarm_batch"] is None:
            st["alarm_batch"] = j
            st["alarm_component"] = ("both" if comp_alarm["primary"] and comp_alarm["secondary"] else
                                     "primary" if comp_alarm["primary"] else "secondary")
            first_event = first_event or ("REFUSED:SHIFT_ALARM", j)
        # label audit (labels of batch j arrive after its unlabelled data were monitored)
        if lab:
            x = _scene_x(rows, lab, tau, floor_c, alpha_e6, n_max)
            if x is not None:
                newly = audit.add(x)
                if newly and st["violation_batch"] is None:
                    st["violation_batch"] = j
                    first_event = first_event or ("VIOLATED", j)
        st["trace"].append({"batch": j, "verdict": first_event[0] if first_event else "VALID",
                            "log_m_e9": [m["primary"]["log_m_e9"], m["secondary"]["log_m_e9"]],
                            "audit_logk_e9": int(round(audit.logk * 1e9)), "n_audited": len(audit.xs)})
    if refusal is not None:
        raise refusal
    if len(monitor_chain) != len(dep_units):
        st["trace"].append({"batch": len(monitor_chain) + 1, "verdict": "REFUSED:CHAIN_BROKEN"})
        raise Refuse("CHAIN_BROKEN", f"{len(dep_units)} deployment batches but "
                                     f"{len(monitor_chain)} monitor receipts")
    if anchor is not None:
        st["anchor"] = check_anchor(anchor, log_pub, g, genv, head_env)
    if audit.xs:
        lb = audit.lower_bound()
        st["audit"] = {"rule": rbcg1.AUDIT_ID, "n_scenes_audited": len(audit.xs),
                       "log_capital_at_alpha_e9": int(round(audit.logk * 1e9)),
                       "log_threshold_e9": int(round(audit.log_thr * 1e9)),
                       "cs_lower_bound_mean_loss_e9": int(math.floor((lb - audit.m0) * 1e9)),
                       "violated_at": audit.violated_at}
    if first_event is None:
        return "VALID", f"guarantee and {st['n_batches']} monitor receipts verified; no alarm, no violation"
    if first_event[0] == "REFUSED:SHIFT_ALARM":
        raise Refuse("SHIFT_ALARM", f"exchangeability monitor alarmed at batch {first_event[1]} "
                                    f"({st['alarm_component']})")
    return "VIOLATED", f"label audit: betting CS lower bound on E[L] above 0 at batch {first_event[1]}"


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


def _read_key(path):
    pk = _read(path).decode("ascii").strip()
    return pk + "\n" if pk.startswith("-----BEGIN") else pk


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Fail-closed RBCG v1 verifier (synthetic data only)")
    ap.add_argument("--guarantee", required=True)
    ap.add_argument("--pubkey", required=True, help="PEM file or file holding SEC1 hex")
    ap.add_argument("--cal-rows", required=True, help="JSONL, one row per line")
    ap.add_argument("--monitor-chain", help="JSONL, one envelope per line")
    ap.add_argument("--deploy-rows", help="JSONL")
    ap.add_argument("--labels", help="JSON object {cid: label}")
    ap.add_argument("--policy", help="JSON object overriding DEFAULT_POLICY keys")
    ap.add_argument("--anchor", help="JSON transparency-log anchor (STHs + proofs)")
    ap.add_argument("--log-pubkey", help="PEM of the transparency-log key")
    a = ap.parse_args(argv)
    try:
        res = verify(_read(a.guarantee), _read_key(a.pubkey), _read_rows(a.cal_rows),
                     [ln for ln in _read(a.monitor_chain).splitlines() if ln.strip()]
                     if a.monitor_chain else None,
                     _read_rows(a.deploy_rows) if a.deploy_rows else None,
                     loads_strict(_read(a.labels), "labels") if a.labels else None,
                     loads_strict(_read(a.policy), "policy") if a.policy else None,
                     _read(a.anchor) if a.anchor else None,
                     _read_key(a.log_pubkey) if a.log_pubkey else None)
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
