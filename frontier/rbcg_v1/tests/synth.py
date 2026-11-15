"""synth.py -- deterministic SYNTHETIC rows for the RBCG v1 test suite (no simulator, no real data)."""
import random

import rbcg1

CL = ("cholinergic", "cyanide", "pulmonary", "vesicant", "opioid", "none")


def synth_rows(seed, n_units, unit_size=24, prefix="cal", b_a=0.5, b_b=8.0, err_scale=1.6,
               err_add=0.0, with_label=True, ood_rate=0.15, inf_rate=0.98, ood_unit_rate=None):
    """1 - pmax ~ (5/6) Beta(b_a, b_b); P(wrong | pmax) = min(1, err_scale * (1 - pmax) + err_add).
    With ood_unit_rate set, a unit is 'OOD-heavy' with that probability and then each row is OOD
    with probability 0.8 (else 0.02); otherwise each row is OOD with probability ood_rate."""
    rng = random.Random(seed)
    rows, labels = [], {}
    for u in range(n_units):
        uid = f"{prefix}/U{u:04d}"
        p_ood = ood_rate if ood_unit_rate is None else (0.8 if rng.random() < ood_unit_rate else 0.02)
        for i in range(unit_size):
            pm = 1.0 - (5.0 / 6.0) * rng.betavariate(b_a, b_b)
            pred = CL[rng.randrange(6)]
            wrong = rng.random() < min(1.0, err_scale * (1.0 - pm) + err_add)
            lab = CL[(CL.index(pred) + 1 + rng.randrange(5)) % 6] if wrong else pred
            r = {"cid": f"{uid}/C{i:02d}", "unit": uid, "inf": rng.random() < inf_rate,
                 "ood": rng.random() < p_ood, "pmax_e12": min(rbcg1.E12, int(pm * rbcg1.E12)),
                 "pred": pred}
            if with_label:
                r["label"] = lab
            labels[r["cid"]] = lab
            rows.append(r)
    return rows, labels


def crafted_units(prefix, n_units, pmax, n_wrong_per_unit, unit_size=24, ood=False):
    """Deterministic deployment units: every row informative, at a fixed pmax; the first
    n_wrong_per_unit rows of each unit are mislabelled."""
    rows, labels = [], {}
    for u in range(n_units):
        uid = f"{prefix}/U{u:04d}"
        for i in range(unit_size):
            cid = f"{uid}/C{i:02d}"
            rows.append({"cid": cid, "unit": uid, "inf": True, "ood": ood,
                         "pmax_e12": int(pmax * rbcg1.E12), "pred": "opioid"})
            labels[cid] = "none" if i < n_wrong_per_unit else "opioid"
    return rows, labels


BINDINGS = {"model_id": "synthetic-test-model-v1", "model_sha256": "a" * 64,
            "weights_file_sha256": "b" * 64, "advisory_code_sha256": "c" * 64,
            "governance_code_sha256": "d" * 64, "sim_code_sha256": "e" * 64}
PROTOCOL_SHA = "f" * 64
