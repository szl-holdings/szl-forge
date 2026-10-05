# Model Lab grouped dependency repair — 2026-10-05

## Invariant and demonstrated failure

This proposal derives from PR #558 at `9b7ff1708693e0ff872f6e78a38a3af6d093bfd3`. Hosted Python 3.11 and 3.12 jobs `111758160643` and `111758160815` fail because Model Lab requires FastAPI <0.142 while the update constrains 0.142.2. Additional parent metadata establishes two next conflicts: Pydantic 2.13.5 requires exactly core 2.46.5, and SymPy 1.14.0 requires mpmath <1.4. The grouped constraints instead selected 2.49.0 and 1.4.1. The changed Torch pin would also replace the workflow's explicitly installed 2.14.0+cpu build with PyPI 2.14.1 and its Linux CUDA dependency closure.

The invariant is one satisfiable test closure with parent-mandated transitive versions and the explicit CPU Torch build retained. Production/GPU environment qualification is separate.

## Repair

Allow FastAPI 0.142 patch releases while keeping the <0.143 bound. Retain the separately installed 2.14.0+cpu Torch build, restore core 2.46.5 and mpmath 1.3.0, and freeze the new FastAPI OpenTelemetry API requirement at the observed 1.45.0. Other compatible grouped updates are preserved. Comments distinguish proposed version changes from historical resolver evidence.

## Observed local receipts

A fresh isolated Python 3.12.14 environment installed `torch==2.14.0+cpu` from the official CPU index and then the repaired Model Lab test constraints. Installation and `python -m pip check` pass. Native Torch reports exactly 2.14.0+cpu and `torch.version.cuda is None`.

The complete Model Lab suite passes: **387 passed, 2 skipped, 12 subtests passed**. Both unchanged skips require native Windows filesystem metadata. Source/API/UI negative publication contracts remain enforced. The repository's workflow and README CPU installation commands are unchanged. `git diff --check` passes.

## Remaining qualification

Local Python is 3.12.14. Fresh hosted Python 3.11/3.12 jobs and the other root repository gates must pass before normal admission. Windows-native skipped behavior is not locally established. These are version constraints rather than artifact hash locks, and neither GPU training, owner-machine state, production rollout nor signed release is claimed. The API-created source commit is unsigned; normal GitHub protected merge rules still apply.
