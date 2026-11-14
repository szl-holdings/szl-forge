#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1.py -- receipt-bound conformal guarantee, version 1 (RBCG v1), for selective prediction.

IN-SILICO RESEARCH ONLY. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE.
This module must never receive clinical or patient data. Python 3.10+ standard library only.

Changes from v0 (see METHOD_v1.md):
  * LTT p-values are computed at the SCENE level (the exchangeable unit) on the linearized
    selective loss L_s(tau) = (e_s - alpha * a_s) / n_max, shifted to X_s = L_s + alpha in [0, 1].
    H0(tau): E[X] >= alpha, which is equivalent to selective risk >= alpha.
    Primary p-value: one-sided predictable plug-in betting capital (Waudby-Smith & Ramdas style),
    p = 1 / max_t K_t (Ville). Secondary: Hoeffding-Bentkus on the same scene means.
    Fixed-sequence testing over the pre-specified grid is unchanged.
  * Label audit: anytime-valid betting lower confidence sequence on E[L] at tau*; VIOLATED when
    alpha is excluded, i.e. the lower bound on E[L] exceeds 0. Replaces Clopper-Pearson + union bound.
  * Monitor: two conformal test martingales, pre-registered primary score = per-scene OOD count,
    secondary = v0 score mean(1 - pmax); Bonferroni split of delta_s (Ville + union bound).
  * Receipts v1 bind the protocol digest and the transparency-log id (see rbcg1_log.py).
Sections 1 (exact binomial, CP, Wilson) and 4 (canonical JSON, DSSE PAE, ECDSA P-256 with RFC 6979)
are carried over from v0 unchanged except for the v1 payload types.
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

RBCG_VERSION = "1.0.0"
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
# 2. LTT fixed-sequence calibration with SCENE-LEVEL betting p-values (v1)
# =============================================================================================
ROW_SCHEMA = "szl.rbcg.row/v0"
CAL_ROW_KEYS = ("cid", "inf", "label", "ood", "pmax_e12", "pred", "unit")
DEP_ROW_KEYS = ("cid", "inf", "ood", "pmax_e12", "pred", "unit")
GRID_C_DEFAULT = tuple(range(99, 29, -1))          # tau = 0.99 .. 0.30, most conservative first
OOD_FLOOR_C_DEFAULT = 81                            # frozen tau_ood of the TRIAGE-X advisory
SELECTOR_ID = "tau_c_ood_floor_v0"
PROCEDURE_ID = "ltt_fixed_sequence_scene_betting_v1"
LOSS_ID = "scene_linearized_selective_loss_v1"      # X = (e - alpha * a) / n_max + alpha in [0, 1]
PVALUE_RULE = "wsr_prpl_onesided_betting_maxcapital_ceil_e9_v1"
SECONDARY_PVALUE_RULE = "hoeffding_bentkus_scene_mean_ceil_e9_v1"
UNIT_ROWS_MAX_DEFAULT = 24
BET_C_E3_DEFAULT = 750                              # truncation c = 3/4 (WSR suggest 1/2 or 3/4)
MIN_ACCEPTED_NOMINAL = 30


def accept_threshold_e12(row, tau_c, ood_floor_c):
    """pmax threshold (in 1e-12 units) for a row: tau, or max(tau, floor) under the OOD condition."""
    return (max(tau_c, ood_floor_c) if row["ood"] else tau_c) * 10 ** 10


def accepts(row, tau_c, ood_floor_c):
    """Selector family tau_c_ood_floor_v0 (unchanged from v0)."""
    return bool(row["inf"]) and row["pmax_e12"] >= accept_threshold_e12(row, tau_c, ood_floor_c)


def ceil_e9(p):
    """Conservative fixed-point encoding of a probability: ceil(p * 1e9)."""
    return int(math.ceil(p * 1e9))


def loss_num(a, e, alpha_e6, n_max):
    """Exact numerator of X = (e - alpha a)/n_max + alpha over the denominator n_max * 10**6."""
    if not (0 <= e <= a <= n_max):
        raise ValueError("need 0 <= errors <= accepted <= n_max")
    return e * 10 ** 6 - alpha_e6 * a + alpha_e6 * n_max


def unit_counts_by_tau(units, grid_c, ood_floor_c, labels=None):
    """Fast path. For each tau in grid order, the list over units of (accepted a_s, errors e_s).
    Labels come from the rows ('label') unless a {cid: label} dict is given. Each row has a
    maximal accepting tau (pmax // 1e10, capped at 101); per-unit suffix sums give a_s(tau)."""
    per_unit = []
    for _, rows in units:
        hn, hk = [0] * 103, [0] * 103
        for r in rows:
            if not r["inf"]:
                continue
            pm = r["pmax_e12"]
            if r["ood"] and pm < ood_floor_c * 10 ** 10:
                continue
            tm = min(pm // 10 ** 10, 101)
            lab = r["label"] if labels is None else labels[r["cid"]]
            hn[tm] += 1
            hk[tm] += r["pred"] != lab
        for u in range(100, -1, -1):
            hn[u] += hn[u + 1]
            hk[u] += hk[u + 1]
        per_unit.append((hn, hk))
    return [[(hn[t], hk[t]) for hn, hk in per_unit] for t in grid_c]


def prpl_lambdas(xs, delta):
    """Predictable plug-in bet sizes before truncation, one-sided (log(1/delta)):
    lambda_t = sqrt(2 log(1/delta) / (sigma2_{t-1} * t * log(1 + t))), with
    mu_t = (1/2 + sum_{i<=t} x_i)/(t + 1) and sigma2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2)/(t + 1)."""
    out, mu_sum, var_sum = [], 0.5, 0.25
    two_log = 2.0 * math.log(1.0 / delta)
    for t, x in enumerate(xs, start=1):
        out.append(math.sqrt(two_log / ((var_sum / t) * t * math.log1p(t))))
        mu_t = (mu_sum + x) / (t + 1)
        var_sum += (x - mu_t) ** 2
        mu_sum += x
    return out


def bet_log_kmax(xs, m, delta, c):
    """log max_{0<=t<=n} K_t for H0: E[X] >= m, K_t = prod_{i<=t} (1 + lambda_i (m - x_i)),
    lambda_i = min(prpl_i, c / (1 - m)) in [0, c/(1-m)], so every factor is >= 1 - c > 0."""
    cap = c / (1.0 - m)
    logk = best = 0.0
    for lam, x in zip(prpl_lambdas(xs, delta), xs):
        logk += math.log1p(min(lam, cap) * (m - x))
        if logk > best:
            best = logk
    return best


def bet_pvalue(xs, m, delta, c):
    """Anytime-valid p-value min(1, 1 / max_t K_t): super-uniform under H0 by Ville's inequality."""
    return min(1.0, math.exp(-bet_log_kmax(xs, m, delta, c)))


def _h1(a, b):
    if a <= 0.0:
        return -math.log1p(-b)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def hb_pvalue(nums, den, alpha_e6):
    """Hoeffding-Bentkus p-value for H0: E[X] >= alpha with n = len(nums) scene means X = num/den:
    min(exp(-n h1(min(mu, alpha), alpha)), e * BinomCDF(ceil(n mu); n, alpha)), ceil taken exactly."""
    n = len(nums)
    if n == 0:
        return 1.0
    alpha = alpha_e6 / 1e6
    s = sum(nums)
    mu = s / (n * den)
    hoeff = math.exp(-n * _h1(min(mu, alpha), alpha))
    kc = -((-s) // den)                              # exact ceil(n * mu)
    bent = math.e * binom_cdf(min(kc, n), n, alpha)
    return min(1.0, hoeff, bent)


def ltt_table(units, grid_c, alpha_e6, delta_e6, ood_floor_c, n_max=UNIT_ROWS_MAX_DEFAULT,
              c_e3=BET_C_E3_DEFAULT, counts=None):
    """Per-tau scene-level table in grid order: n_units, n_acc, k_err, sum of loss numerators, the
    betting log max-capital (floor e9), the betting p-value and the HB p-value (ceil e9)."""
    if counts is None:
        counts = unit_counts_by_tau(units, grid_c, ood_floor_c)
    alpha, delta, c = alpha_e6 / 1e6, delta_e6 / 1e6, c_e3 / 1000.0
    den = n_max * 10 ** 6
    table = []
    for t, cs in zip(grid_c, counts):
        nums = [loss_num(a, e, alpha_e6, n_max) for a, e in cs]
        logk = bet_log_kmax([v / den for v in nums], alpha, delta, c)
        table.append({"tau_c": t, "n_units": len(cs), "n_acc": sum(a for a, _ in cs),
                      "k_err": sum(e for _, e in cs), "sum_x_num": sum(nums),
                      "logk_e9": int(math.floor(logk * 1e9)),
                      "p_bet_e9": ceil_e9(min(1.0, math.exp(-logk))),
                      "p_hb_e9": ceil_e9(hb_pvalue(nums, den, alpha_e6))})
    return table


def ltt_fixed_sequence(table, delta_e6, key="p_bet_e9"):
    """Fixed-sequence testing over the pre-ordered grid: reject H0(tau) while p <= delta and stop
    at the first non-rejection. Returns (tau_star_c or None, number rejected). FWER <= delta."""
    tau_star, n_rej = None, 0
    for e in table:
        if e[key] <= delta_e6 * 1000:
            tau_star, n_rej = e["tau_c"], n_rej + 1
        else:
            break
    return tau_star, n_rej


def calibrate(rows, alpha_e6, delta_e6, grid_c=GRID_C_DEFAULT, ood_floor_c=OOD_FLOOR_C_DEFAULT,
              n_max=UNIT_ROWS_MAX_DEFAULT, c_e3=BET_C_E3_DEFAULT, units=None):
    units = group_units(rows) if units is None else units
    for _, ur in units:
        if len(ur) > n_max:
            raise ValueError(f"unit has {len(ur)} rows > n_max {n_max}")
    table = ltt_table(units, grid_c, alpha_e6, delta_e6, ood_floor_c, n_max, c_e3)
    tau_star, n_rej = ltt_fixed_sequence(table, delta_e6, "p_bet_e9")
    tau_hb, n_rej_hb = ltt_fixed_sequence(table, delta_e6, "p_hb_e9")
    return {"table": table, "table_sha256": sha256_hex(canonical_json(table)), "tau_star_c": tau_star,
            "n_rejected": n_rej, "secondary_tau_star_c": tau_hb, "secondary_n_rejected": n_rej_hb,
            "n_units": len(units)}


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
# 3. Online exchangeability monitor (label-free): two conformal test martingales (v1)
# =============================================================================================
MONITOR_PRIMARY_SCORE_ID = "unit_ood_count_v1"                 # number of rows with ood == True
MONITOR_SECONDARY_SCORE_ID = "unit_mean_one_minus_pmax_e12_v0"  # the v0 score
MONITOR_PVALUE_ID = "smoothed_conformal_vovk2003_v0"
MONITOR_MARTINGALE_ID = "mixture_power_martingale_v0"
MONITOR_COMBINATION_ID = "bonferroni_union_two_ctm_v1"
MONITOR_UNIT = "scene"
EPSILONS_E3_DEFAULT = (100, 200, 300, 400, 500, 600, 700, 800, 900)
DELTA_S_PRIMARY_E6_DEFAULT = 7500
DELTA_S_SECONDARY_E6_DEFAULT = 2500
THETA_DOMAIN_PRIMARY = "szl.rbcg.monitor.theta.primary/v1"
THETA_DOMAIN_SECONDARY = "szl.rbcg.monitor.theta.secondary/v1"


def unit_score_ood_count(rows):
    """Primary score: the number of rows of the unit flagged with the advisory's OOD condition."""
    if not rows:
        raise ValueError("empty unit")
    return sum(1 for r in rows if r["ood"])


def unit_score_conf(rows):
    """Secondary score (v0): floor(mean over the unit of (1 - pmax)) in 1e-12 units."""
    if not rows:
        raise ValueError("empty unit")
    return sum(E12 - r["pmax_e12"] for r in rows) // len(rows)


SCORE_FUNCS = {MONITOR_PRIMARY_SCORE_ID: unit_score_ood_count,
               MONITOR_SECONDARY_SCORE_ID: unit_score_conf}


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


def theta_seed_for(domain, cal_rows_sha256):
    return hashlib.sha256((domain + "|" + cal_rows_sha256).encode("ascii")).hexdigest()


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


def _logsumexp_list(vals):
    m = max(vals)
    return m + math.log(math.fsum(math.exp(v - m) for v in vals))


class CTM:
    """One conformal test martingale over units. Calibration scores are the initial reference;
    betting (power martingales mixed uniformly over epsilons) starts at the first deployment unit.
    Alarm (latched) once log M >= log(1/delta)."""

    def __init__(self, cal_scores, epsilons_e3, delta_e6, theta_seed):
        self.ref = sorted(cal_scores)
        self.eps = [e / 1000.0 for e in epsilons_e3]
        self.log_comp = [0.0] * len(self.eps)
        self.j = 0
        self.log_thr = log_alarm_threshold(delta_e6)
        self.alarmed = False
        self.seed = theta_seed

    def step(self, score):
        self.j += 1
        th = theta(self.seed, self.j)
        p = smoothed_p(self.ref, score, th)
        lp = math.log(p)
        for i, e in enumerate(self.eps):
            self.log_comp[i] += math.log(e) + (e - 1.0) * lp
        log_m = _logsumexp_list(self.log_comp) - math.log(len(self.eps))
        if log_m >= self.log_thr:
            self.alarmed = True
        bisect.insort(self.ref, score)
        return {"j": self.j, "score": score, "theta": th, "p": p, "log_m": log_m, "alarm": self.alarmed}


class TwoScoreMonitor:
    """Primary (OOD count) and secondary (mean 1 - pmax) CTMs with a Bonferroni split of delta_s:
    alarm once either crosses its own threshold. P(any alarm) <= delta_A + delta_B (Ville, union)."""

    def __init__(self, cal_units, mon):
        self.comp = {}
        for name in ("primary", "secondary"):
            spec = mon[name]
            f = SCORE_FUNCS[spec["score_id"]]
            self.comp[name] = (f, CTM([f(rows) for _, rows in cal_units], mon["epsilons_e3"],
                                      spec["delta_s_e6"], spec["theta_seed"]))
        self.alarmed = False

    def step(self, rows):
        out = {}
        for name, (f, ctm) in self.comp.items():
            out[name] = ctm.step(f(rows))
        self.alarmed = self.alarmed or out["primary"]["alarm"] or out["secondary"]["alarm"]
        out["alarm"] = self.alarmed
        return out


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
PT_GUARANTEE = "application/vnd.szl.rbcg.guarantee.v1+json"
PT_MONITOR = "application/vnd.szl.rbcg.monitor.v1+json"
PT_STH = "application/vnd.szl.rbcg.sth.v1+json"
PT_PROTOCOL = "application/vnd.szl.rbcg.protocol.v1+json"


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
# 5. Receipts (v1)
# =============================================================================================
GUARANTEE_SCHEMA = "szl.rbcg.guarantee/v1"
MONITOR_SCHEMA = "szl.rbcg.monitor/v1"
ANCHOR_LEAF_RULE = "rfc9162_sha256_leaf_of_canonical_envelope_v1"
AUDIT_ID = "wsr_prpl_betting_cs_lower_v1"
DELTA_AUDIT_E6_DEFAULT = 50000


class NoGuarantee(Exception):
    """LTT rejected no hypothesis: no guarantee may be issued (fail closed)."""


def rows_sha256(rows):
    return sha256_hex(canonical_json(rows))


def fmt_c(v):
    return f"{v // 100}.{v % 100:02d}"


def fmt_e6(v):
    s = f"{v // 10 ** 6}.{v % 10 ** 6:06d}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def claim_text(model_id, tau_star_c, ood_floor_c, alpha_e6, delta_e6, n_units, n_rows, seed_family,
               delta_s_primary_e6, delta_s_secondary_e6):
    ds = delta_s_primary_e6 + delta_s_secondary_e6
    return (f"RBCG v1 guarantee (in-silico, synthetic data only; not for clinical use). For model "
            f"{model_id} with abstention thresholds tau={fmt_c(tau_star_c)} and "
            f"tau_ood=max(tau,{fmt_c(ood_floor_c)}): with probability at least "
            f"{fmt_e6(10 ** 6 - delta_e6)} over the draw of the {n_units} calibration scenes "
            f"({n_rows} rows; seed family {seed_family}), the selective error P(prediction wrong | "
            f"advisory accepted) is at most {fmt_e6(alpha_e6)} on deployment scenes exchangeable with "
            f"the calibration scenes. Void once the bound exchangeability monitor alarms (delta_s="
            f"{fmt_e6(ds)}: primary {fmt_e6(delta_s_primary_e6)}, secondary {fmt_e6(delta_s_secondary_e6)}).")


def rbcg_code_sha256():
    return sha256_file(os.path.abspath(__file__))


def monitor_spec(cal_rows_sha, n_cal_units, epsilons_e3=EPSILONS_E3_DEFAULT,
                 delta_s_primary_e6=DELTA_S_PRIMARY_E6_DEFAULT,
                 delta_s_secondary_e6=DELTA_S_SECONDARY_E6_DEFAULT):
    return {"primary": {"score_id": MONITOR_PRIMARY_SCORE_ID, "delta_s_e6": delta_s_primary_e6,
                        "theta_seed": theta_seed_for(THETA_DOMAIN_PRIMARY, cal_rows_sha)},
            "secondary": {"score_id": MONITOR_SECONDARY_SCORE_ID, "delta_s_e6": delta_s_secondary_e6,
                          "theta_seed": theta_seed_for(THETA_DOMAIN_SECONDARY, cal_rows_sha)},
            "p_value_id": MONITOR_PVALUE_ID, "martingale_id": MONITOR_MARTINGALE_ID,
            "combination": MONITOR_COMBINATION_ID, "unit": MONITOR_UNIT,
            "epsilons_e3": list(epsilons_e3), "n_cal_units": n_cal_units,
            "betting_start": "first_deployment_unit"}


def issue_guarantee(cal_rows, *, key, bindings, seed_family, env_profile, log_id, protocol_sha256,
                    alpha_e6=100000, delta_e6=100000, grid_c=GRID_C_DEFAULT, ood_floor_c=OOD_FLOOR_C_DEFAULT,
                    n_max=UNIT_ROWS_MAX_DEFAULT, c_e3=BET_C_E3_DEFAULT, epsilons_e3=EPSILONS_E3_DEFAULT,
                    delta_s_primary_e6=DELTA_S_PRIMARY_E6_DEFAULT,
                    delta_s_secondary_e6=DELTA_S_SECONDARY_E6_DEFAULT,
                    delta_audit_e6=DELTA_AUDIT_E6_DEFAULT, audit_c_e3=BET_C_E3_DEFAULT,
                    issued_at_utc="", cal=None):
    """Run scene-level LTT on calibration rows and return (signed guarantee envelope, info).
    Raises NoGuarantee when no tau is certified. `bindings` must hold model_id, model_sha256,
    weights_file_sha256, advisory_code_sha256, governance_code_sha256 and sim_code_sha256."""
    units = group_units(cal_rows)
    if cal is None:
        cal = calibrate(cal_rows, alpha_e6, delta_e6, grid_c, ood_floor_c, n_max, c_e3, units=units)
    if cal["tau_star_c"] is None:
        raise NoGuarantee("no hypothesis rejected; no guarantee issued")
    rh = rows_sha256(cal_rows)
    b = dict(bindings)
    b["rbcg_code_sha256"] = rbcg_code_sha256()
    payload = {
        "schema": GUARANTEE_SCHEMA, "rbcg_version": RBCG_VERSION,
        "bindings": b, "protocol_sha256": protocol_sha256,
        "calibration": {"seed_family": seed_family, "env_profile": env_profile, "n_rows": len(cal_rows),
                        "n_units": len(units), "rows_sha256": rh, "row_schema": ROW_SCHEMA,
                        "unit_rows_max": n_max},
        "procedure": {"id": PROCEDURE_ID, "loss": LOSS_ID, "alpha_e6": alpha_e6, "delta_e6": delta_e6,
                      "grid_c": list(grid_c), "grid_order": "descending",
                      "selector": {"id": SELECTOR_ID, "ood_floor_c": ood_floor_c},
                      "p_value": PVALUE_RULE, "bet_c_e3": c_e3,
                      "secondary_p_value": SECONDARY_PVALUE_RULE,
                      "table_sha256": cal["table_sha256"], "n_rejected": cal["n_rejected"],
                      "tau_star_c": cal["tau_star_c"],
                      "secondary_n_rejected": cal["secondary_n_rejected"],
                      "secondary_tau_star_c": cal["secondary_tau_star_c"]},
        "monitor": monitor_spec(rh, len(units), epsilons_e3, delta_s_primary_e6, delta_s_secondary_e6),
        "audit": {"id": AUDIT_ID, "delta_audit_e6": delta_audit_e6, "bet_c_e3": audit_c_e3,
                  "unit": MONITOR_UNIT, "loss": LOSS_ID},
        "anchoring": {"log_id": log_id, "leaf_rule": ANCHOR_LEAF_RULE},
        "claim": claim_text(b["model_id"], cal["tau_star_c"], ood_floor_c, alpha_e6, delta_e6, len(units),
                            len(cal_rows), seed_family, delta_s_primary_e6, delta_s_secondary_e6),
        "issued_at_utc": issued_at_utc,
        "synthetic_only": True, "not_for_clinical_use": True,
    }
    env = dsse_envelope(payload, PT_GUARANTEE, key)
    return env, {"payload": payload, "table": cal["table"], "digest": envelope_digest(env), "cal": cal}


def _comp(s):
    return {"score": s["score"], "p_e12": int(round(s["p"] * E12)), "log_m_e9": int(round(s["log_m"] * 1e9)),
            "alarm": s["alarm"]}


class MonitorChain:
    """Issues one signed, hash-chained MONITOR receipt per deployment unit (scene)."""

    def __init__(self, guarantee_env, cal_rows, key):
        g = json.loads(base64.b64decode(guarantee_env["payload"]))
        self.g_digest = envelope_digest(guarantee_env)
        self.prev = self.g_digest
        self.key = key
        self.seq = 0
        self.head = None
        self.monitor = TwoScoreMonitor(group_units(cal_rows), g["monitor"])

    def add_unit(self, unit_id, rows, deploy_source):
        st = self.monitor.step(rows)
        self.seq += 1
        payload = {"schema": MONITOR_SCHEMA, "guarantee_sha256": self.g_digest, "seq": self.seq,
                   "prev_sha256": self.prev, "unit_id": unit_id, "batch_sha256": rows_sha256(rows),
                   "batch_n_rows": len(rows), "deploy_source": deploy_source,
                   "primary": _comp(st["primary"]), "secondary": _comp(st["secondary"]),
                   "alarm": st["alarm"]}
        env = dsse_envelope(payload, PT_MONITOR, self.key)
        self.prev = envelope_digest(env)
        self.head = env
        return env, st


# =============================================================================================
# 6. Label audit: anytime-valid betting lower confidence sequence (v1)
# =============================================================================================
def scene_loss(rows, labels, tau_c, ood_floor_c, alpha_e6, n_max):
    """(X, a, e) of one labelled deployment scene at tau, or None when an accepted row has no label."""
    a = e = 0
    for r in rows:
        if accepts(r, tau_c, ood_floor_c):
            lab = labels.get(r["cid"])
            if lab is None:
                return None
            a += 1
            e += r["pred"] != lab
    return loss_num(a, e, alpha_e6, n_max) / (n_max * 10 ** 6), a, e


class BettingAudit:
    """Lower confidence sequence for mu = E[X], X = L + alpha in [0, 1], by the predictable plug-in
    betting capital K_t(m) = prod_{i<=t} (1 + min(lambda_i, c/m) (X_i - m)). Every factor is
    >= 1 - c > 0. CS_t = {m : K_t(m) < 1/delta}; K_t(m) is non-increasing in m, so CS_t = [l_t, 1].
    VIOLATED at the first look t with K_t(alpha) >= 1/delta, i.e. alpha not in CS_t, i.e. the
    lower bound on E[L] = mu - alpha is at least 0. P(ever VIOLATED | E[L] <= 0) <= delta (Ville)."""

    def __init__(self, alpha_e6, delta_audit_e6, c_e3):
        self.m0 = alpha_e6 / 1e6
        self.delta = delta_audit_e6 / 1e6
        self.c = c_e3 / 1000.0
        self.log_thr = math.log(1.0 / self.delta)
        self.xs, self.lams = [], []
        self._mu_sum, self._var_sum = 0.5, 0.25
        self.logk = 0.0
        self.violated_at = None

    def add(self, x):
        t = len(self.xs) + 1
        lam = math.sqrt(2.0 * self.log_thr / ((self._var_sum / t) * t * math.log1p(t)))
        self.lams.append(lam)
        self.xs.append(x)
        self.logk += math.log1p(min(lam, self.c / self.m0) * (x - self.m0))
        if self.violated_at is None and self.logk >= self.log_thr:
            self.violated_at = t
        mu_t = (self._mu_sum + x) / (t + 1)
        self._var_sum += (x - mu_t) ** 2
        self._mu_sum += x
        return self.violated_at is not None

    def log_capital(self, m):
        """log K_t(m) for 0 < m <= 1 over all looks so far."""
        cap = self.c / m
        return math.fsum(math.log1p(min(l, cap) * (x - m)) for l, x in zip(self.lams, self.xs))

    def lower_bound(self, iters=60):
        """l_t = inf{m in (0, 1] : K_t(m) < 1/delta} by bisection (returns the lower end, i.e. a
        conservative bound); 0 if even m -> 0+ is in the CS."""
        if not self.xs:
            return 0.0
        lo, hi = 1e-12, 1.0
        if self.log_capital(lo) < self.log_thr:
            return 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.log_capital(mid) >= self.log_thr:
                lo = mid
            else:
                hi = mid
        return lo
