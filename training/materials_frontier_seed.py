#!/usr/bin/env python3
"""Five materials-grand-challenge domains -> first-party *toy* SFT candidate.

NO contest benchmarks, scraped articles, copyrighted figures, training weights,
GPU jobs, rights admission, publisher action or hidden evaluation data.
A deterministic arithmetic oracle makes original pedagogy, not scientific
predictions or proof of solving any of the five real open problems.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from math import isqrt
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "szl.materials-frontier-synthetic-sft/v1"
MANIFEST_SCHEMA = "szl.materials-frontier-candidate-manifest/v1"
SYSTEM = (
    "Answer the explicitly synthetic mathematical exercise. State the units, "
    "the checked invariant and the evidence boundary. A toy relation is not "
    "a verified theory, experimental result, solved grand challenge or "
    "qualified scientific model. Refuse unsupported research claims."
)
SPLITS = {
    "train": tuple(range(1, 17)),
    "dev": tuple(range(101, 105)),
    "diagnostic": tuple(range(201, 205)),
}
CHALLENGES = {
    "cuprate": {
        "title": "Predictive theory of cuprate superconductivity",
        "url": "https://grandchallenges.discoveredmaterials.com/cuprate-superconductivity/",
        "missing": "Independent Tc(p), d-wave phase, pseudogap and transport predictions across cuprate families",
    },
    "interface_electrical": {
        "title": "Interfacial electrical properties from processing history",
        "url": "https://grandchallenges.discoveredmaterials.com/interfacial-electrical-properties/",
        "missing": "Process-history-dependent thickness, composition, Dit(E) and Qf across interface families",
    },
    "quantum_chemistry": {
        "title": "Breaking the speed-accuracy barrier in quantum chemistry",
        "url": "https://grandchallenges.discoveredmaterials.com/speed-accuracy-barrier/",
        "missing": "Exchange-correlation functional, GMTKN55/higher-level error and compute-time comparisons",
    },
    "driven_materials": {
        "title": "Tuning materials with light",
        "url": "https://grandchallenges.discoveredmaterials.com/tuning-materials-with-light/",
        "missing": "Converged nonequilibrium electron-electron, electron-phonon, phonon-phonon dynamics and limits",
    },
    "heat_interfaces": {
        "title": "Predicting heat transport across solid interfaces",
        "url": "https://grandchallenges.discoveredmaterials.com/interfacial-heat-transport/",
        "missing": "Measured transferable G(T), AMM/DMM baselines, held-out interfaces and uncertainty",
    },
}
# User screenshot defines interesting CAPABILITY TARGETS, not verified scores
# or lawful training/derivative-use rights in proprietary benchmark items.
BENCHMARK_CAPABILITIES = {
    "scientific_terminal_agent": ("Terminal-Bench-Science", "Agents' Last Exam"),
    "scientific_method": ("Vals Science Index",),
    "coding_maintenance": ("FrontierCode 2.0", "ModelDial"),
    "cad_and_3d": ("BenchCAD 2.1 Core", "AA-3D"),
    "broad_reasoning": ("Artificial Analysis Intelligence Index",),
    "professional_legal": ("Harvey LAB-AA",),
    "professional_business": ("GDPval v2",),
}
TASKS = ("calculate", "invariant", "refusal")
DOMAIN_ORDER = tuple(CHALLENGES)
MAX_FILE = 2_000_000
SKILLS = {
    "calculate": "dimensioned analytical bookkeeping",
    "invariant": "symmetry, conservation and consistency checking",
    "refusal": "scientific abstention and evidence sufficiency",
}


class CurriculumHold(ValueError):
    """Stable source/admission refusal; no provider content or secrets."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CurriculumHold(code)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fmt(value: Fraction | int) -> str:
    q = Fraction(value)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _physics(domain: str, skill: str, case: int) -> tuple[str, str, dict]:
    """Return original toy input, exact oracle response and parameters."""
    require(domain in CHALLENGES and skill in TASKS, "UNKNOWN_DOMAIN_OR_TASK")
    require(type(case) is int and 1 <= case <= 250, "CASE_OUTSIDE_BOUNDS")
    label = f"Synthetic exercise {domain}-{skill}-{case}. "
    if domain == "cuprate":
        phases = ("0", "pi/2", "pi")
        cos = {"0": 1, "pi/2": 0, "pi": -1}
        kx, ky = phases[case % 3], phases[(case // 3 + 1) % 3]
        delta = 4 * (6 + case % 13)
        gap = Fraction(delta * (cos[kx] - cos[ky]), 2)
        if skill == "calculate":
            prompt = (
                label + "For the explicitly toy d-wave expression "
                "Delta(kx,ky)=(Delta0/2)*(cos(kx)-cos(ky)), "
                f"Delta0={delta} meV, kx={kx}, ky={ky}. "
                "Compute Delta; does this establish a microscopic Tc theory?"
            )
            answer = (
                f"Delta={fmt(gap)} meV. The trigonometric values were exact. "
                "This d-wave toy calculation does not predict Tc, the pseudogap or experimental data."
            )
        elif skill == "invariant":
            prompt = (
                label + f"Take Delta0={delta} meV, kx={kx}, ky={ky}. "
                "Under the toy d-wave expression, compute both swapped gaps "
                "and test Delta(ky,kx)=-Delta(kx,ky)."
            )
            answer = (
                f"Delta(kx,ky)={fmt(gap)} meV; Delta(ky,kx)={fmt(-gap)} meV. "
                "Antisymmetry holds algebraically; no cuprate phase-diagram claim follows."
            )
        else:
            family = ("LSCO", "YBCO", "Bi-family", "Hg-family")[case % 4]
            prompt = (
                label + f"A draft fits Delta0={delta} meV for an illustrative "
                f"{family} gap curve only. Can one claim the five-family "
                "cuprate grand challenge is solved?"
            )
            answer = (
                "HOLD. A fitted gap curve alone does not establish a consistent "
                "microscopic mechanism or independent Tc(p), magnetic, pseudogap "
                "and transport predictions. Report fit data separately from truly predicted data."
            )
        fields = {"delta0_mev": delta, "kx": kx, "ky": ky}
    elif domain == "interface_electrical":
        x = 8 + 2 * (case % 19)
        a = 4 + 2 * (case % 9)
        b = 2  # nm^2/min
        t = x * (x + a) // b
        discriminant = a * a + 4 * b * t
        require(
            isqrt(discriminant) ** 2 == discriminant
            and (isqrt(discriminant) - a) // 2 == x,
            "DEAL_GROVE_ORACLE_INCONSISTENT",
        )
        if skill == "calculate":
            prompt = (
                label + "In a purely illustrative Deal-Grove growth law "
                "x^2+A*x=B*t (units: x,A nm; B nm^2/min; t min), "
                f"A={a}, B={b}, t={t}. Find the nonnegative oxide thickness x."
            )
            answer = (
                f"x={x} nm (positive quadratic root). This controlled toy law "
                "does not predict real interface traps Dit(E) or fixed oxide charge Qf."
            )
        elif skill == "invariant":
            x2 = x + 2
            t2 = x2 * (x2 + a) // b
            prompt = (
                label + f"With A={a} nm and B={b} nm^2/min, "
                f"toy growth reports x1={x} nm and x2={x2} nm. "
                "Compute times t1,t2 from x^2+A*x=B*t and test monotonicity."
            )
            answer = (
                f"t1={t} min and t2={t2} min. t2>t1 so x(t) increases "
                "under the assumed positive parameters; no physical chamber was validated."
            )
        else:
            interface = ("Si/SiO2", "SiC/SiO2", "HfO2/SiO2/Si", "Ge/GeO2")[case % 4]
            prompt = (
                label + f"A synthetic {interface} run reports oxide thickness "
                f"{x} nm but no electrical measurement, composition profile "
                "or fabrication history. Can Dit(E) and Qf be declared predicted?"
            )
            answer = (
                "HOLD. Thickness alone is insufficient. Independently bind "
                "orientation, temperature/ambient process history, composition "
                "and measured Dit(E)/Qf before claiming electrical-quality prediction."
            )
        fields = {"A_nm": a, "B_nm2_per_min": b, "time_min": t, "x_nm": x}
    elif domain == "quantum_chemistry":
        ea, eb = -140 - case % 41, -65 - case % 17
        ec, ed = -175 - (2 * case) % 47, -55 - case % 29
        delta = ec + ed - ea - eb
        if skill == "calculate":
            prompt = (
                label + "A synthetic reaction A+B->C+D has toy electronic energies "
                f"EA={ea}, EB={eb}, EC={ec}, ED={ed} kJ/mol. Compute "
                "DeltaE=(EC+ED)-(EA+EB). Is this a CCSD(T) result?"
            )
            answer = (
                f"DeltaE={delta} kJ/mol. These are invented arithmetic energies, "
                "not measured or CCSD(T)-calculated reaction/activation energies."
            )
        elif skill == "invariant":
            e1 = eb - ea
            e2 = ec - eb
            direct = ec - ea
            prompt = (
                label + f"For toy levels EA={ea}, EB={eb}, EC={ec} kJ/mol, "
                "compare DeltaE(A->B)+DeltaE(B->C) to DeltaE(A->C). "
                "Check Hess-cycle closure."
            )
            answer = (
                f"DeltaE_AB={e1}, DeltaE_BC={e2}, sum={direct} kJ/mol; "
                f"DeltaE_AC={direct} kJ/mol. Cycle closes by construction, "
                "not by chemical validation."
            )
        else:
            prompt = (
                label + f"A proposed XC functional has one toy reaction "
                f"DeltaE={delta} kJ/mol but no independent benchmarks or "
                "timing. Does this establish a superior functional?"
            )
            answer = (
                "HOLD. Require a specified functional, reproducible implementation, "
                "GMTKN55/W(T)MAD-style error comparison under the correct definition, "
                "independent reactions and measured CPU/GPU runtimes. "
                "One arithmetic reaction cannot establish improvement."
            )
        fields = {"EA_kj_mol": ea, "EB_kj_mol": eb, "EC_kj_mol": ec, "ED_kj_mol": ed}
    elif domain == "driven_materials":
        e0 = 100 + case % 29
        work = 16 + 2 * (case % 19)
        diss = 5 + case % 11
        final = e0 + work - diss
        if skill == "calculate":
            prompt = (
                label + f"A toy pulse-energy ledger starts E0={e0} meV/unit-cell, "
                f"receives W={work} meV/unit-cell, and loses Q={diss} meV/unit-cell. "
                "What is Efinal from energy conservation?"
            )
            answer = (
                f"Efinal={final} meV/unit-cell. This bookkeeping alone "
                "does not simulate non-equilibrium Green functions, excitons or lattice dynamics."
            )
        elif skill == "invariant":
            no_drive = e0 - diss
            prompt = (
                label + f"In a toy closed-accounting step, E0={e0} and "
                f"dissipated energy Q={diss} meV/unit-cell, with W=0. "
                "What upper bound follows for Efinal and why?"
            )
            answer = (
                f"Efinal={no_drive} meV/unit-cell <= E0={e0}. "
                "Without an external source, positive dissipation cannot increase "
                "energy in this accounting model; physical scattering was not calculated."
            )
        else:
            prompt = (
                label + f"A toy optical pulse deposits W={work} meV/unit-cell "
                "but has no electron–electron, electron–phonon, phonon–phonon "
                "coupling results or equilibrium limit. Is an ultrafast materials "
                "grand challenge solved?"
            )
            answer = (
                "HOLD. Verify coupled nonequilibrium dynamics, energy conservation, "
                "equilibrium/weak-drive recovery, reproducible model-system references "
                "and independent material-specific time traces before any such claim."
            )
        fields = {
            "E0_mev_cell": e0,
            "pump_work_mev_cell": work,
            "dissipation_mev_cell": diss,
        }
    else:
        g = 40 + 10 * (case % 15)  # MW/(m^2 K)
        flux = 5 + case % 21  # MW/m^2
        jump = Fraction(flux, g)
        if skill == "calculate":
            prompt = (
                label + f"An illustrative interface has conductance G={g} "
                "MW/(m^2 K) and steady heat flux q="
                f"{flux} MW/m^2. Compute DeltaT=q/G and label units."
            )
            answer = (
                f"DeltaT={fmt(jump)} K. This uses an assigned G; it does not "
                "predict G(T) for a real interface or outperform AMM/DMM."
            )
        elif skill == "invariant":
            prompt = (
                label + f"A toy interface has G1={g} MW/(m^2 K), "
                f"G2={2 * g} MW/(m^2 K). Give the exact ratio RK2/RK1 "
                "with RK=1/G and check the direction."
            )
            answer = (
                "RK2/RK1=1/2; doubling G halves interface thermal resistance "
                "at the same stated conditions. This is algebra, not a measured interface gain."
            )
        else:
            interface = ("Al/Si", "Al/GaN", "Cu/diamond", "Au/PMMA/Si")[case % 4]
            prompt = (
                label + f"A simulation assumes G={g} MW/(m^2 K) for "
                f"{interface} but has no measured G(T), source uncertainty "
                "or independent interface samples. Can transferable accuracy be claimed?"
            )
            answer = (
                "HOLD. Require provenance for temperature-dependent measurements, "
                "AMM/DMM and atomistic baselines, uncertainty, and evaluation on "
                "interfaces excluded from parameter fitting. An assumed conductance "
                "does not prove a predictive method."
            )
        fields = {"G_mw_m2_k": g, "q_mw_m2": flux}
    require(
        "HOLD" in answer if skill == "refusal" else "HOLD" not in answer,
        "REFUSAL_LABEL_CONTRACT",
    )
    return prompt, answer, fields


def rows(split: str) -> list[dict]:
    require(split in SPLITS, "SPLIT_NOT_RECOGNIZED")
    items = []
    for domain in DOMAIN_ORDER:
        for skill in TASKS:
            for case in SPLITS[split]:
                prompt, answer, fields = _physics(domain, skill, case)
                identity = sha(canonical([domain, skill, case, split]))
                items.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": {
                            "schema": SCHEMA,
                            "split": split,
                            "challenge": domain,
                            "task": skill,
                            "skill": SKILLS[skill],
                            "case_id": identity,
                            "parameters": fields,
                            "review_state": "CANDIDATE_ONLY",
                            "source_kind": "FIRST_PARTY_SYNTHETIC_TOY",
                            "training_authority": "NONE",
                            "scientific_prediction_verified": False,
                            "independent_holdout": False,
                            "official_benchmark_item": False,
                        },
                    }
                )
    return items


def _validator():
    path = ROOT / "tools" / "validate_sft_dataset.py"
    require(path.is_file(), "FORGE_SFT_VALIDATOR_MISSING")
    spec = importlib.util.spec_from_file_location("forge_materials_sft_validator", path)
    require(spec is not None and spec.loader is not None, "VALIDATOR_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_dataset


def stage(output: Path) -> dict:
    """Materialize into a NEW private directory; no network or model import."""
    output = Path(output)
    require(
        output.is_absolute()
        and output.parent.is_dir()
        and not output.exists()
        and not output.is_symlink()
        and not output.parent.is_symlink()
        and not output.resolve().is_relative_to(ROOT.resolve()),
        "NEW_OUTSIDE_REPO_DIRECTORY_REQUIRED",
    )
    work = Path(tempfile.mkdtemp(prefix=".szl-materials-stage-", dir=output.parent))
    try:
        count = {}
        seen = set()
        for split in SPLITS:
            items = rows(split)
            require(
                len(items) == (len(CHALLENGES) * len(TASKS) * len(SPLITS[split])),
                "SPLIT_COUNT_CHANGED",
            )
            for item in items:
                meta = item["metadata"]
                require(meta["case_id"] not in seen, "CROSS_SPLIT_ID_OVERLAP")
                seen.add(meta["case_id"])
            body = b"".join(canonical(item) + b"\n" for item in items)
            require(0 < len(body) <= MAX_FILE, "CORPUS_BYTE_BUDGET")
            name = split + ".jsonl"
            (work / name).write_bytes(body)
            count[split] = {"rows": len(items), "sha256": sha(body), "bytes": len(body)}
        validate = _validator()
        report = validate(
            str(work / "train.jsonl"),
            min_examples=32,
            disjoint_from=[str(work / "dev.jsonl"), str(work / "diagnostic.jsonl")],
        )
        require(
            report["status"] == "VALID"
            and report["records"] == count["train"]["rows"]
            and report["duplicate_inputs"] == 0
            and report["duplicate_conversations"] == 0
            and report["cross_dataset_overlaps"] == 0
            and all(r["status"] == "VALID" for r in report["reference_datasets"]),
            "FORGE_SFT_VALIDATION_HOLD",
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "producer_source_sha256": sha(Path(__file__).read_bytes()),
            "training_authority": "NONE",
            "training_launch": False,
            "model_published": False,
            "contest_submission": False,
            "review_state": "CANDIDATE_ONLY",
            "scientific_claim": "NOT_VERIFIED_TOY_EXERCISES_ONLY",
            "data_rights": "FIRST_PARTY_GENERATED_TEXT_REQUIRES_INTERNAL_REVIEW",
            "external_papers_or_benchmark_items_copied": False,
            "independent_holdout": False,
            "holdout_statement": "DEV_AND_DIAGNOSTIC_REUSE_SAME_GENERATOR_NOT_INDEPENDENT",
            "split_group_disjointness": "CASE_IDENTITIES_AND_INPUTS_ONLY",
            "challenge_references": CHALLENGES,
            "capability_benchmarks_source": "USER_SCREENSHOT_UNVERIFIED_COMPARISONS_NOT_TRAINING_ITEMS",
            "capability_evaluation_dimensions": {
                k: list(v) for k, v in BENCHMARK_CAPABILITIES.items()
            },
            "files": count,
            "sft_validation": {
                "schema": report["schema"],
                "records": report["records"],
                "duplicate_inputs": report["duplicate_inputs"],
                "cross_dataset_overlaps": report["cross_dataset_overlaps"],
                "references": [
                    {
                        "records": r["records"],
                        "overlapping_inputs": r["overlapping_inputs"],
                    }
                    for r in report["reference_datasets"]
                ],
            },
            "independent_external_scientific_evaluation": "NOT_STARTED",
            "curriculum_not_benchmark_performance": True,
            "official_competition_hidden_tests_used": False,
            "salad_spend_authorized": False,
            "massed_gpu_launch_authorized": False,
            "huggingface_publication_authorized": False,
            "signed": False,
        }
        (work / "manifest.json").write_bytes(
            json.dumps(
                manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
            ).encode()
            + b"\n"
        )
        # Verify immutable artifact bytes before exposing complete directory.
        for filename, meta in count.items():
            buf = (work / (filename + ".jsonl")).read_bytes()
            require(
                len(buf) == meta["bytes"] and sha(buf) == meta["sha256"],
                "SOURCE_FILE_WRITEBACK_CHANGED",
            )
        os.replace(work, output)
        return manifest
    finally:
        if work.exists():
            shutil.rmtree(work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = stage(args.output_dir)
    except (CurriculumHold, OSError) as exc:
        code = str(exc) if isinstance(exc, CurriculumHold) else type(exc).__name__
        print(json.dumps({"status": "HOLD", "reason": code}))
        return 2
    print(
        json.dumps(
            {
                "status": "CANDIDATE_ONLY_NOT_ADMITTED_FOR_TRAINING",
                "schema": result["schema"],
                "counts": {name: row["rows"] for name, row in result["files"].items()},
                "source_sha256": result["producer_source_sha256"],
                "sft_validation": result["sft_validation"],
                "model_training_started": False,
                "model_publication": False,
                "contest_submission": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
