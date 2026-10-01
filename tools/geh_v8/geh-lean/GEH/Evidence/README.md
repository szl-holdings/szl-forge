GEH/Evidence is NOT a build target of the compliance library (see lakefile.toml
globs = ["GEH.Compliance.+"]). Files here must start with `-- GEH-EVIDENCE-ONLY`.
Property-based testing (`plausible`, `slim_check`) produces EVIDENCE_ONLY
artifacts; the harness classifier refuses them in the compliance lane.
