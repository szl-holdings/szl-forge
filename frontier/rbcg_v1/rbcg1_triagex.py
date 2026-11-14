#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1_triagex.py -- adapter (v1; identical logic to v0) from the SZL TRIAGE-X in-silico stack to RBCG rows.

SYNTHETIC DATA ONLY. Every casualty is a simulator record. NOT FOR CLINICAL USE.
Uses the frozen advisory (weights + ADVISORY_FREEZE.json) unchanged; never retrains, never
re-selects its frozen thresholds, and never reads the frozen test/shifted splits except through
load_frozen_shifted(), which v1 never calls
(the frozen splits are SPENT; v1 E3 only re-verifies the stored v0 E3 rows).
"""
import os
import sys
import zipfile

import rbcg1 as rbcg


class Stack:
    def __init__(self, stack_dir):
        self.dir = os.path.abspath(stack_dir)
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
        import triagex_sim as sim
        import triagex_advisory as adv
        import triagex_governance as gov
        self.sim, self.adv, self.gov = sim, adv, gov
        with open(os.path.join(self.dir, adv.WEIGHTS_FILE), "rb") as fh:
            w = gov.loads_strict(fh.read())
        with open(os.path.join(self.dir, adv.FREEZE_FILE), "rb") as fh:
            self.freeze = gov.loads_strict(fh.read())
        self.advisor = adv.Advisor(w, self.freeze["tau"], self.freeze["tau_ood"])
        adv.require_freeze(self.dir, self.advisor)          # refuses unless the freeze binds these weights
        self.classes = adv.CLASSES
        self.bindings = {
            "model_id": adv.MODEL_ID,
            "model_sha256": self.advisor.sha256,
            "weights_file_sha256": rbcg.sha256_file(os.path.join(self.dir, adv.WEIGHTS_FILE)),
            "advisory_code_sha256": rbcg.sha256_file(adv.__file__),
            "governance_code_sha256": rbcg.sha256_file(gov.__file__),
            "sim_code_sha256": rbcg.sha256_file(sim.__file__),
        }

    def _row(self, family, o):
        p = self.advisor.probs(o)
        cl = self.classes
        best = max(cl, key=lambda c: (p[c], -cl.index(c)))
        return {"cid": f"{family}/{o['casualty_id']}", "unit": f"{family}/{o['scene_id']}",
                "inf": bool(self.adv.has_informative_signs(o)), "ood": bool(self.adv.ood_condition(o)),
                "pmax_e12": int(round(p[best] * rbcg.E12)), "pred": best}

    def scene(self, family, idx, env_profile, first_scene=1, with_label=True):
        """One fresh scene of a NAMED seed family (scene seed = sha256(family|index) per the
        simulator's SEED_RULE); every observation is validated before use."""
        sid = f"S{first_scene + idx:04d}"
        sc = self.sim.generate_scene(sid, self.sim.scene_seed(family, idx), env_profile)
        self.sim.check_scene(sc)
        rows, labels = [], {}
        for o, t in zip(sc["observations"], sc["truth"]):
            v = self.gov.validate_observation(o)
            if not v["ok"]:
                raise RuntimeError(f"observation refused: {v['code']}")
            r = self._row(family, o)
            if with_label:
                r["label"] = t["toxidrome"]
            labels[r["cid"]] = t["toxidrome"]
            rows.append(r)
        return rows, labels, sc["observations"], sc["truth"]

    def family(self, family, n_scenes, env_profile, first_scene=1, keep_obs=False, with_label=True):
        """n_scenes fresh scenes of a NAMED seed family -> (rows, labels, obs, truth)."""
        rows, labels, obs_all, truth_all = [], {}, [], []
        for idx in range(n_scenes):
            r, lab, o, t = self.scene(family, idx, env_profile, first_scene, with_label)
            rows += r
            labels.update(lab)
            if keep_obs:
                obs_all += o
                truth_all += t
        return rows, labels, obs_all, truth_all

    def load_frozen_shifted(self, workdir):
        """REFUSED in v1: the frozen test and shifted splits are SPENT (scored once by v0 E3)."""
        raise RuntimeError("REFUSED: the frozen splits are spent; RBCG v1 never re-scores them")
