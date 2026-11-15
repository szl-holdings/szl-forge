#!/usr/bin/env python3
# RBCG v0 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg.py -- receipt-bound conformal guarantee (RBCG v0) for selective prediction.

IN-SILICO RESEARCH ONLY. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE.
This module must never receive clinical or patient data. Python 3.10+ standard library only.

Contents
  1. Exact binomial tails in log space (math.lgamma); Clopper-Pearson and Wilson intervals.
  2. Learn-then-Test (LTT) fixed-sequence calibration of a selective-risk threshold.
  3. Label-free online exchangeability monitor: smoothed conformal p-values over calibration
     units then deployment units, a mixture of power martingales, alarm at 1/delta_s.
  4. Canonical JSON, DSSE v1 PAE, ECDSA P-256 / SHA-256 with RFC 6979 nonces (pure Python;
     NOT constant-time; for throwaway test keys only, never for real keys).
  5. GUARANTEE and MONITOR receipt payloads, DSSE envelopes and hash chaining.

Every statistical statement depends on the assumptions in METHOD.md (exchangeability of
calibration and deployment data; validity of the binomial test conditional on n_accept).
"""
from __future__ import annotations

import base64
import bisect
import hashlib
import hmac
import json
import math
import os
import re
import secrets

RBCG_VERSION = "0.1.0"
E12 = 10 ** 12

# =============================================================================================
# 1. Binomial tails, Clopper-Pearson and Wilson bounds
# =============================================================================================


def _require_int(name, v, lo=None, hi=None):
    if type(v) is not int:
        raise TypeError(f"{name} must be int, got {type(v).__name__}")
    if lo is not None and v < lo:
        raise ValueError(f"{name}={v} < {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{name}={v} > {hi}")


def _require_prob(name, p, open_ends=False):
    if type(p) not in (int, float):
        raise TypeError(f"{name} must be a real number, got {type(p).__name__}")
    p = float(p)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"{name}={p!r} outside [0, 1]")
    if open_ends and (p <= 0.0 or p >= 1.0):
        raise ValueError(f"{name}={p!r} must lie strictly inside (0, 1)")
    return p


def log_binom_pmf(j, n, p):
    """log P[X = j] for X ~ Binomial(n, p), via math.lgamma (exact up to rounding)."""
    if p == 0.0:
        return 0.0 if j == 0 else -math.inf
    if p == 1.0:
        return 0.0 if j == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            + j * math.log(p) + (n - j) * math.log1p(-p))


def _logsumexp(vals):
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(v - m) for v in vals))


def binom_cdf(k, n, p):
    """P[X <= k] for X ~ Binomial(n, p): the sum over j = 0..k, in log space."""
    _require_int("n", n, 0)
    _require_int("k", k)
    p = _require_prob("p", p)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, math.exp(_logsumexp([log_binom_pmf(j, n, p) for j in range(0, k + 1)])))


def binom_sf(k, n, p):
    """P[X >= k] for X ~ Binomial(n, p): the sum over j = k..n, in log space."""
    _require_int("n", n, 0)
    _require_int("k", k)
    p = _require_prob("p", p)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, math.exp(_logsumexp([log_binom_pmf(j, n, p) for j in range(k, n + 1)])))


def cp_lower(k, n, delta):
    """One-sided Clopper-Pearson lower bound at level 1 - delta: the p with P_p[X >= k] = delta
    (0 when k = 0). Bisection; the lower end of the final bracket is returned (conservative)."""
    _require_int("n", n, 0)
    _require_int("k", k, 0, n)
    delta = _require_prob("delta", delta, open_ends=True)
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if binom_sf(k, n, mid) < delta:
            lo = mid
        else:
            hi = mid
    return lo


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound at level 1 - delta: the p with P_p[X <= k] = delta
    (1 when k = n). Bisection; the upper end of the final bracket is returned (conservative)."""
    _require_int("n", n, 0)
    _require_int("k", k, 0, n)
    delta = _require_prob("delta", delta, open_ends=True)
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def cp_interval(k, n, conf=0.95):
    d = (1.0 - conf) / 2.0
    return cp_lower(k, n, d), cp_upper(k, n, d)


def wilson_interval(k, n, conf=0.95):
    """Wilson score interval for a binomial proportion (Wilson 1927)."""
    _require_int("n", n, 0)
    _require_int("k", k, 0, n)
    if n == 0:
        return 0.0, 1.0
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf) / 2.0)
    ph, z2 = k / n, z * z
    den = 1.0 + z2 / n
    center = (ph + z2 / (2.0 * n)) / den
    half = z * math.sqrt(ph * (1.0 - ph) / n + z2 / (4.0 * n * n)) / den
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return lo, hi


# =============================================================================================
# 2. Learn-then-Test fixed-sequence calibration of the selective-risk threshold
# =============================================================================================
ROW_SCHEMA = "szl.rbcg.row/v0"
CAL_ROW_KEYS = ("cid", "inf", "label", "ood", "pmax_e12", "pred", "unit")
DEP_ROW_KEYS = ("cid", "inf", "ood", "pmax_e12", "pred", "unit")
GRID_C_DEFAULT = tuple(range(99, 29, -1))          # tau = 0.99 .. 0.30, most conservative first
OOD_FLOOR_C_DEFAULT = 81                            # frozen tau_ood of the TRIAGE-X advisory
SELECTOR_ID = "tau_c_ood_floor_v0"
PROCEDURE_ID = "ltt_fixed_sequence_binomial_v0"
PVALUE_RULE = "binom_cdf_k_n_alpha_ceil_e9"
MIN_ACCEPTED_NOMINAL = 30                           # the frozen method's MIN_ACCEPTED_FOR_TAU


def accept_threshold_e12(row, tau_c, ood_floor_c):
    """pmax threshold (in 1e-12 units) for a row: tau, or max(tau, floor) under the OOD condition."""
    return (max(tau_c, ood_floor_c) if row["ood"] else tau_c) * 10 ** 10


def accepts(row, tau_c, ood_floor_c):
    """Selector family tau_c_ood_floor_v0: INSUFFICIENT_SIGNS rows (inf False) always abstain;
    OOD-condition rows need pmax >= max(tau, floor); all other rows need pmax >= tau."""
    return bool(row["inf"]) and row["pmax_e12"] >= accept_threshold_e12(row, tau_c, ood_floor_c)


def ceil_e9(p):
    """Conservative fixed-point encoding of a probability: ceil(p * 1e9)."""
    return int(math.ceil(p * 1e9))


def ltt_table(rows, grid_c, alpha_e6, ood_floor_c):
    """Per-tau (n accepted, k errors among accepted, p_e9) in grid order.
    p tests H0: selective risk(tau) > alpha with p = BinomCDF(k; n, alpha); p = 1 when n = 0.
    Fast path: each row has a maximal accepting tau; the verifier recomputes directly."""
    alpha = alpha_e6 / 1e6
    hist_n, hist_k = [0] * 102, [0] * 102
    for r in rows:
        if not r["inf"]:
            continue
        pm = r["pmax_e12"]
        if r["ood"] and pm < ood_floor_c * 10 ** 10:
            continue
        tm = min(pm // 10 ** 10, 101)
        hist_n[tm] += 1
        if r["pred"] != r["label"]:
            hist_k[tm] += 1
    suf_n, suf_k = [0] * 103, [0] * 103
    for u in range(101, -1, -1):
        suf_n[u] = suf_n[u + 1] + hist_n[u]
        suf_k[u] = suf_k[u + 1] + hist_k[u]
    table = []
    for t in grid_c:
        n, k = suf_n[t], suf_k[t]
        p = binom_cdf(k, n, alpha) if n > 0 else 1.0
        table.append({"k": k, "n": n, "p_e9": ceil_e9(p), "tau_c": t})
    return table


def ltt_fixed_sequence(table, delta_e6):
    """Fixed-sequence testing over the pre-ordered grid: reject H0(tau) while p <= delta and stop
    at the first non-rejection. Returns (tau_star_c or None, number rejected). FWER <= delta."""
    tau_star, n_rej = None, 0
    for e in table:
        if e["p_e9"] <= delta_e6 * 1000:
            tau_star, n_rej = e["tau_c"], n_rej + 1
        else:
            break
    return tau_star, n_rej


def nominal_tau_c(rows, grid_c, alpha_e6, ood_floor_c, min_accepted=MIN_ACCEPTED_NOMINAL):
    """The nominal rule (what the frozen advisory does), for comparison only: the least
    conservative tau whose calibration selective error k/n <= alpha with n >= min_accepted."""
    for t in sorted(grid_c):
        n = k = 0
        for r in rows:
            if accepts(r, t, ood_floor_c):
                n += 1
                k += r["pred"] != r["label"]
        if n >= min_accepted and k * 10 ** 6 <= alpha_e6 * n:
            return t
    return None


def calibrate(rows, alpha_e6, delta_e6, grid_c=GRID_C_DEFAULT, ood_floor_c=OOD_FLOOR_C_DEFAULT):
    table = ltt_table(rows, grid_c, alpha_e6, ood_floor_c)
    tau_star, n_rej = ltt_fixed_sequence(table, delta_e6)
    return {"table": table, "table_sha256": sha256_hex(canonical_json(table)),
            "tau_star_c": tau_star, "n_rejected": n_rej}


def selective_counts(rows, labels, tau_c, ood_floor_c):
    """(n_rows, n_accepted, errors among labelled accepted, labelled accepted)."""
    n_acc = k = n_lab = 0
    for r in rows:
        if accepts(r, tau_c, ood_floor_c):
            n_acc += 1
            lab = labels.get(r["cid"]) if labels is not None else r.get("label")
            if lab is not None:
                n_lab += 1
                k += r["pred"] != lab
    return len(rows), n_acc, k, n_lab


# =============================================================================================
# 3. Online exchangeability monitor (label-free)
# =============================================================================================
MONITOR_SCORE_ID = "unit_mean_one_minus_pmax_e12_v0"
MONITOR_PVALUE_ID = "smoothed_conformal_vovk2003_v0"
MONITOR_MARTINGALE_ID = "mixture_power_martingale_v0"
MONITOR_UNIT = "scene"
EPSILONS_E3_DEFAULT = (100, 200, 300, 400, 500, 600, 700, 800, 900)
DELTA_S_E6_DEFAULT = 10000
THETA_DOMAIN = "szl.rbcg.monitor.theta/v0"


def unit_score(rows):
    """Nonconformity score of one unit (scene): floor(mean over its rows of (1 - pmax)) in 1e-12
    units. Label-free; larger means stranger (less confident)."""
    if not rows:
        raise ValueError("empty unit")
    return sum(E12 - r["pmax_e12"] for r in rows) // len(rows)


def group_units(rows):
    """Split rows into consecutive units by their 'unit' key; a unit id may not recur later."""
    units, seen, cur = [], set(), None
    for r in rows:
        u = r["unit"]
        if cur is None or u != cur[0]:
            if u in seen:
                raise ValueError(f"unit {u!r} recurs non-consecutively")
            seen.add(u)
            cur = (u, [])
            units.append(cur)
        cur[1].append(r)
    return units


def theta_seed_for(cal_rows_sha256):
    return hashlib.sha256((THETA_DOMAIN + "|" + cal_rows_sha256).encode("ascii")).hexdigest()


def theta(seed_hex, j):
    """Tie-breaking uniform for deployment unit j (1-based), in (0, 1]; fixed at issuance."""
    d = hashlib.sha256(f"{seed_hex}|{j}".encode("ascii")).digest()
    u = int.from_bytes(d[:7], "big") >> 3
    return (u + 1) / 9007199254740992.0     # 2**53


def smoothed_p(sorted_ref, score, th):
    """Smoothed conformal p-value of `score` against all earlier scores (sorted_ref):
    (#{earlier > score} + th * #{earlier == score, plus itself}) / (len(earlier) + 1)."""
    lo = bisect.bisect_left(sorted_ref, score)
    hi = bisect.bisect_right(sorted_ref, score)
    gt = len(sorted_ref) - hi
    eq = (hi - lo) + 1
    return (gt + th * eq) / (len(sorted_ref) + 1)


def log_alarm_threshold(delta_s_e6):
    return math.log(10 ** 6 / delta_s_e6)


class ExchangeabilityMonitor:
    """Conformal test martingale over units. Calibration unit scores form the initial reference
    multiset; betting (power martingales, mixed uniformly over epsilons) starts at the first
    deployment unit. Alarm (latched) once log M >= log(1/delta_s)."""

    def __init__(self, cal_scores, epsilons_e3=EPSILONS_E3_DEFAULT, delta_s_e6=DELTA_S_E6_DEFAULT,
                 theta_seed=""):
        self.ref = sorted(cal_scores)
        self.eps = [e / 1000.0 for e in epsilons_e3]
        self.log_comp = [0.0] * len(self.eps)
        self.j = 0
        self.log_thr = log_alarm_threshold(delta_s_e6)
        self.alarmed = False
        self.seed = theta_seed

    def step(self, score):
        self.j += 1
        th = theta(self.seed, self.j)
        p = smoothed_p(self.ref, score, th)
        lp = math.log(p)
        for i, e in enumerate(self.eps):
            self.log_comp[i] += math.log(e) + (e - 1.0) * lp
        log_m = _logsumexp(self.log_comp) - math.log(len(self.eps))
        if log_m >= self.log_thr:
            self.alarmed = True
        bisect.insort(self.ref, score)
        return {"j": self.j, "score": score, "theta": th, "p": p, "log_m": log_m,
                "alarm": self.alarmed}


# =============================================================================================
# 4. Canonical JSON, DSSE PAE, ECDSA P-256 / SHA-256 with RFC 6979 deterministic nonces
# =============================================================================================


def canonical_json(obj):
    """SZL A11oy convention (sorted keys, compact separators, UTF-8, ensure_ascii False) plus
    allow_nan=False, so NaN/Infinity can never be serialised."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pae(payload_type, body):
    """DSSE v1 Pre-Authentication Encoding over UTF-8(payload_type) and body bytes:
    "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body, LEN = decimal byte length."""
    if type(payload_type) is not str or type(body) is not bytes:
        raise TypeError("pae(str, bytes) required")
    t = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(t)).encode("ascii") + b" " + t + b" " + \
        str(len(body)).encode("ascii") + b" " + body


# --- NIST P-256 (FIPS 186-4 / SEC 2 secp256r1); constants cross-checked in the test suite
# --- against OpenSSL's explicit prime256v1 parameters and the RFC 6979 A.2.5 key pair.
P256_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
P256_A = P256_P - 3
P256_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
P256_GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
P256_GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
_INF = (1, 1, 0)                                   # Jacobian point at infinity (Z = 0)


def _jdouble(P):
    X1, Y1, Z1 = P
    if Z1 == 0 or Y1 == 0:
        return _INF
    p = P256_P
    delta = Z1 * Z1 % p
    gamma = Y1 * Y1 % p
    beta = X1 * gamma % p
    alpha = 3 * (X1 - delta) * (X1 + delta) % p
    X3 = (alpha * alpha - 8 * beta) % p
    Z3 = ((Y1 + Z1) * (Y1 + Z1) - gamma - delta) % p
    Y3 = (alpha * (4 * beta - X3) - 8 * gamma * gamma) % p
    return (X3, Y3, Z3)


def _jadd(P, Q):
    X1, Y1, Z1 = P
    X2, Y2, Z2 = Q
    if Z1 == 0:
        return Q
    if Z2 == 0:
        return P
    p = P256_P
    Z1Z1 = Z1 * Z1 % p
    Z2Z2 = Z2 * Z2 % p
    U1 = X1 * Z2Z2 % p
    U2 = X2 * Z1Z1 % p
    S1 = Y1 * Z2 * Z2Z2 % p
    S2 = Y2 * Z1 * Z1Z1 % p
    H = (U2 - U1) % p
    r = 2 * (S2 - S1) % p
    if H == 0:
        return _jdouble(P) if r == 0 else _INF
    I = 4 * H * H % p
    J = H * I % p
    V = U1 * I % p
    X3 = (r * r - J - 2 * V) % p
    Y3 = (r * (V - X3) - 2 * S1 * J) % p
    Z3 = ((Z1 + Z2) * (Z1 + Z2) - Z1Z1 - Z2Z2) * H % p
    return (X3, Y3, Z3)


def _to_affine(P):
    X, Y, Z = P
    if Z == 0:
        return None
    p = P256_P
    zi = pow(Z, -1, p)
    zi2 = zi * zi % p
    return (X * zi2 % p, Y * zi2 * zi % p)


def _scalar_mult(k, point):
    """k * point (affine in, affine out; None = infinity). 4-bit fixed window, Jacobian.
    Not constant-time."""
    if k % P256_N == 0 or point is None:
        return None
    k %= P256_N
    base = (point[0], point[1], 1)
    table = [_INF, base]
    for _ in range(14):
        table.append(_jadd(table[-1], base))
    Q = _INF
    for shift in range(252, -1, -4):
        Q = _jdouble(_jdouble(_jdouble(_jdouble(Q))))
        nib = (k >> shift) & 15
        if nib:
            Q = _jadd(Q, table[nib])
    return _to_affine(Q)


def on_curve(x, y):
    if not (0 <= x < P256_P and 0 <= y < P256_P):
        return False
    return (y * y - (x * x * x + P256_A * x + P256_B)) % P256_P == 0


G = (P256_GX, P256_GY)


def _bits2int(b, qlen=256):
    x = int.from_bytes(b, "big")
    blen = len(b) * 8
    return x >> (blen - qlen) if blen > qlen else x


def _int2octets(x, rlen=32):
    return x.to_bytes(rlen, "big")


def rfc6979_nonces(d, h1, q=P256_N):
    """RFC 6979 section 3.2 candidate nonces k (HMAC-SHA-256), yielded in order."""
    qlen = q.bit_length()
    rlen = (qlen + 7) // 8
    x = _int2octets(d, rlen)
    bo = _int2octets(_bits2int(h1, qlen) % q, rlen)
    V = b"\x01" * 32
    K = b"\x00" * 32
    K = hmac.new(K, V + b"\x00" + x + bo, hashlib.sha256).digest()
    V = hmac.new(K, V, hashlib.sha256).digest()
    K = hmac.new(K, V + b"\x01" + x + bo, hashlib.sha256).digest()
    V = hmac.new(K, V, hashlib.sha256).digest()
    while True:
        T = b""
        while len(T) < rlen:
            V = hmac.new(K, V, hashlib.sha256).digest()
            T += V
        k = _bits2int(T, qlen)
        if 1 <= k < q:
            yield k
        K = hmac.new(K, V + b"\x00", hashlib.sha256).digest()
        V = hmac.new(K, V, hashlib.sha256).digest()


def ecdsa_sign_digest(d, h1):
    """ECDSA P-256 signature (r, s) over the SHA-256 digest h1 with RFC 6979 nonces."""
    e = _bits2int(h1, 256)
    for k in rfc6979_nonces(d, h1):
        R = _scalar_mult(k, G)
        r = R[0] % P256_N
        if r == 0:
            continue
        s = pow(k, -1, P256_N) * (e + r * d) % P256_N
        if s == 0:
            continue
        return r, s
    raise RuntimeError("unreachable")


def ecdsa_verify_digest(pub, h1, r, s):
    """True iff (r, s) is a valid ECDSA P-256 signature over digest h1 for public point pub."""
    if type(r) is not int or type(s) is not int:
        return False
    if not (1 <= r < P256_N and 1 <= s < P256_N):
        return False
    if pub is None or not on_curve(pub[0], pub[1]):
        return False
    e = _bits2int(h1, 256)
    w = pow(s, -1, P256_N)
    u1, u2 = e * w % P256_N, r * w % P256_N
    A, B = _scalar_mult(u1, G), _scalar_mult(u2, pub)
    if A is None and B is None:
        return False
    if A is None:
        X = B
    elif B is None:
        X = A
    else:
        X = _to_affine(_jadd((A[0], A[1], 1), (B[0], B[1], 1)))
    if X is None:
        return False
    return X[0] % P256_N == r


# --- DER (signature) and SubjectPublicKeyInfo encodings --------------------------------------
def _der_int(v):
    b = v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")
    if b[0] & 0x80:
        b = b"\x00" + b
    return b"\x02" + bytes([len(b)]) + b


def der_encode_sig(r, s):
    body = _der_int(r) + _der_int(s)
    return b"\x30" + bytes([len(body)]) + body


def der_decode_sig(sig):
    """Strict DER ECDSA-Sig-Value decode: short-form lengths, minimal positive INTEGERs, no
    trailing bytes. Raises ValueError on any deviation."""
    if type(sig) is not bytes or len(sig) < 8 or sig[0] != 0x30 or sig[1] != len(sig) - 2:
        raise ValueError("bad DER SEQUENCE")
    out, i = [], 2
    for _ in range(2):
        if i + 2 > len(sig) or sig[i] != 0x02:
            raise ValueError("bad DER INTEGER tag")
        ln = sig[i + 1]
        if ln == 0 or ln > 33 or i + 2 + ln > len(sig):
            raise ValueError("bad DER INTEGER length")
        v = sig[i + 2:i + 2 + ln]
        if v[0] & 0x80:
            raise ValueError("negative DER INTEGER")
        if ln > 1 and v[0] == 0 and not (v[1] & 0x80):
            raise ValueError("non-minimal DER INTEGER")
        out.append(int.from_bytes(v, "big"))
        i += 2 + ln
    if i != len(sig):
        raise ValueError("trailing bytes after DER signature")
    return out[0], out[1]


# id-ecPublicKey (1.2.840.10045.2.1) and prime256v1 (1.2.840.10045.3.1.7), DER-encoded OIDs
_OID_EC_PUBLIC_KEY = bytes([0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01])
_OID_PRIME256V1 = bytes([0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07])
_SPKI_PREFIX = (b"\x30\x59\x30\x13" + _OID_EC_PUBLIC_KEY + _OID_PRIME256V1 + b"\x03\x42\x00")


class PublicKey:
    def __init__(self, x, y):
        if type(x) is not int or type(y) is not int or not on_curve(x, y):
            raise ValueError("point is not on P-256")
        self.point = (x, y)

    def sec1(self):
        return b"\x04" + self.point[0].to_bytes(32, "big") + self.point[1].to_bytes(32, "big")

    def sec1_hex(self):
        return self.sec1().hex()

    @classmethod
    def from_sec1(cls, b):
        if type(b) is not bytes or len(b) != 65 or b[0] != 4:
            raise ValueError("expected 65-byte uncompressed SEC1 point")
        return cls(int.from_bytes(b[1:33], "big"), int.from_bytes(b[33:], "big"))

    @classmethod
    def from_sec1_hex(cls, h):
        if type(h) is not str or not re.fullmatch(r"04[0-9a-f]{128}", h):
            raise ValueError("expected lowercase hex of an uncompressed SEC1 point")
        return cls.from_sec1(bytes.fromhex(h))

    def spki_der(self):
        return _SPKI_PREFIX + self.sec1()

    @classmethod
    def from_spki_der(cls, der):
        if type(der) is not bytes or len(der) != 91 or der[:26] != _SPKI_PREFIX:
            raise ValueError("not a P-256 SubjectPublicKeyInfo")
        return cls.from_sec1(der[26:])

    def pem(self):
        b = base64.b64encode(self.spki_der()).decode("ascii")
        lines = [b[i:i + 64] for i in range(0, len(b), 64)]
        return "-----BEGIN PUBLIC KEY-----\n" + "\n".join(lines) + "\n-----END PUBLIC KEY-----\n"

    @classmethod
    def from_pem(cls, pem):
        m = re.fullmatch(r"\s*-----BEGIN PUBLIC KEY-----\s*([A-Za-z0-9+/=\s]+?)\s*"
                         r"-----END PUBLIC KEY-----\s*", pem)
        if not m:
            raise ValueError("not a PUBLIC KEY PEM")
        return cls.from_spki_der(base64.b64decode("".join(m.group(1).split()), validate=True))

    def keyid(self):
        """A11oy convention: SHA-256 hex of the normalised SPKI PEM text, stripped."""
        return hashlib.sha256(self.pem().strip().encode("ascii")).hexdigest()

    def verify(self, message, sig_der):
        try:
            r, s = der_decode_sig(sig_der)
        except ValueError:
            return False
        return ecdsa_verify_digest(self.point, hashlib.sha256(message).digest(), r, s)


class SigningKey:
    """Throwaway P-256 signing key. The scalar is never printed, serialised or stored."""

    def __init__(self, d):
        if type(d) is not int or not (1 <= d < P256_N):
            raise ValueError("private scalar out of range")
        self._d = d
        pt = _scalar_mult(d, G)
        self.public = PublicKey(pt[0], pt[1])

    @classmethod
    def generate(cls):
        return cls(secrets.randbelow(P256_N - 1) + 1)

    def __repr__(self):
        return f"SigningKey(keyid={self.public.keyid()[:16]}..., private=<not shown>)"

    def sign_rs(self, message):
        return ecdsa_sign_digest(self._d, hashlib.sha256(message).digest())

    def sign(self, message):
        """DER-encoded ECDSA-SHA256 signature (the encoding a11oy szl_dsse.py emits)."""
        return der_encode_sig(*self.sign_rs(message))


# --- DSSE envelopes ------------------------------------------------------------------------------
PT_GUARANTEE = "application/vnd.szl.rbcg.guarantee+json"
PT_MONITOR = "application/vnd.szl.rbcg.monitor+json"


def dsse_envelope(payload_obj, payload_type, key):
    body = canonical_json(payload_obj)
    sig = key.sign(pae(payload_type, body))
    return {"payload": base64.b64encode(body).decode("ascii"), "payloadType": payload_type,
            "signatures": [{"keyid": key.public.keyid(), "sig": base64.b64encode(sig).decode("ascii")}]}


def envelope_bytes(env):
    return canonical_json(env)


def envelope_digest(env):
    return sha256_hex(canonical_json(env))


# =============================================================================================
# 5. Receipts
# =============================================================================================
GUARANTEE_SCHEMA = "szl.rbcg.guarantee/v0"
MONITOR_SCHEMA = "szl.rbcg.monitor/v0"
DELTA_AUDIT_E6_DEFAULT = 50000
AUDIT_SPENDING = "delta_audit*6/(pi^2*j^2) at look j"


class NoGuarantee(Exception):
    """LTT rejected no hypothesis: no guarantee may be issued (fail closed)."""


def rows_sha256(rows):
    return sha256_hex(canonical_json(rows))


def fmt_c(v):
    return f"{v // 100}.{v % 100:02d}"


def fmt_e6(v):
    s = f"{v // 10 ** 6}.{v % 10 ** 6:06d}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def claim_text(model_id, tau_star_c, ood_floor_c, alpha_e6, delta_e6, n_cal, seed_family,
               delta_s_e6):
    return (f"RBCG v0 guarantee (in-silico, synthetic data only; not for clinical use). For model "
            f"{model_id} with abstention thresholds tau={fmt_c(tau_star_c)} and "
            f"tau_ood=max(tau,{fmt_c(ood_floor_c)}): with probability at least "
            f"{fmt_e6(10 ** 6 - delta_e6)} over the draw of the {n_cal} calibration rows (seed "
            f"family {seed_family}), the selective error P(prediction wrong | advisory accepted) "
            f"is at most {fmt_e6(alpha_e6)} on data exchangeable with the calibration data. Void "
            f"once the bound exchangeability monitor alarms (delta_s={fmt_e6(delta_s_e6)}).")


def rbcg_code_sha256():
    return sha256_file(os.path.abspath(__file__))


def issue_guarantee(cal_rows, *, key, bindings, seed_family, env_profile, alpha_e6=100000,
                    delta_e6=100000, grid_c=GRID_C_DEFAULT, ood_floor_c=OOD_FLOOR_C_DEFAULT,
                    epsilons_e3=EPSILONS_E3_DEFAULT, delta_s_e6=DELTA_S_E6_DEFAULT,
                    delta_audit_e6=DELTA_AUDIT_E6_DEFAULT, issued_at_utc=""):
    """Run LTT on calibration rows and return (signed guarantee envelope, info). Raises
    NoGuarantee when no tau is certified. `bindings` must hold model_id, model_sha256,
    weights_file_sha256, advisory_code_sha256, governance_code_sha256 and sim_code_sha256."""
    cal = calibrate(cal_rows, alpha_e6, delta_e6, grid_c, ood_floor_c)
    if cal["tau_star_c"] is None:
        raise NoGuarantee("no hypothesis rejected; no guarantee issued")
    rh = rows_sha256(cal_rows)
    units = group_units(cal_rows)
    b = dict(bindings)
    b["rbcg_code_sha256"] = rbcg_code_sha256()
    payload = {
        "schema": GUARANTEE_SCHEMA, "rbcg_version": RBCG_VERSION,
        "bindings": b,
        "calibration": {"seed_family": seed_family, "env_profile": env_profile,
                        "n_rows": len(cal_rows), "n_units": len(units), "rows_sha256": rh,
                        "row_schema": ROW_SCHEMA},
        "procedure": {"id": PROCEDURE_ID, "alpha_e6": alpha_e6, "delta_e6": delta_e6,
                      "grid_c": list(grid_c), "grid_order": "descending",
                      "selector": {"id": SELECTOR_ID, "ood_floor_c": ood_floor_c},
                      "p_value": PVALUE_RULE, "table_sha256": cal["table_sha256"],
                      "n_rejected": cal["n_rejected"], "tau_star_c": cal["tau_star_c"]},
        "monitor": {"score_id": MONITOR_SCORE_ID, "p_value_id": MONITOR_PVALUE_ID,
                    "martingale_id": MONITOR_MARTINGALE_ID, "unit": MONITOR_UNIT,
                    "epsilons_e3": list(epsilons_e3), "delta_s_e6": delta_s_e6,
                    "theta_seed": theta_seed_for(rh), "n_cal_units": len(units),
                    "betting_start": "first_deployment_unit"},
        "audit": {"delta_audit_e6": delta_audit_e6, "spending": AUDIT_SPENDING},
        "claim": claim_text(b["model_id"], cal["tau_star_c"], ood_floor_c, alpha_e6, delta_e6,
                            len(cal_rows), seed_family, delta_s_e6),
        "issued_at_utc": issued_at_utc,
        "synthetic_only": True, "not_for_clinical_use": True,
    }
    env = dsse_envelope(payload, PT_GUARANTEE, key)
    return env, {"payload": payload, "table": cal["table"], "digest": envelope_digest(env)}


class MonitorChain:
    """Issues one signed, hash-chained MONITOR receipt per deployment unit (scene)."""

    def __init__(self, guarantee_env, cal_rows, key):
        g = json.loads(base64.b64decode(guarantee_env["payload"]))
        mon = g["monitor"]
        self.g_digest = envelope_digest(guarantee_env)
        self.prev = self.g_digest
        self.key = key
        self.seq = 0
        self.monitor = ExchangeabilityMonitor(
            [unit_score(rows) for _, rows in group_units(cal_rows)], mon["epsilons_e3"],
            mon["delta_s_e6"], mon["theta_seed"])

    def add_unit(self, unit_id, rows, deploy_source):
        st = self.monitor.step(unit_score(rows))
        self.seq += 1
        payload = {"schema": MONITOR_SCHEMA, "guarantee_sha256": self.g_digest, "seq": self.seq,
                   "prev_sha256": self.prev, "unit_id": unit_id, "batch_sha256": rows_sha256(rows),
                   "batch_n_rows": len(rows), "deploy_source": deploy_source,
                   "score": st["score"], "p_e12": int(round(st["p"] * E12)),
                   "log_m_e9": int(round(st["log_m"] * 1e9)), "alarm": st["alarm"]}
        env = dsse_envelope(payload, PT_MONITOR, self.key)
        self.prev = envelope_digest(env)
        return env, st
