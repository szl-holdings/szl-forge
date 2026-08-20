from __future__ import annotations

import argparse
from contextlib import closing
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if (
    os.environ.get("OAC_TEST_INSTALLED_PACKAGE") != "1"
    and str(SOURCE_ROOT) not in sys.path
):
    sys.path.insert(0, str(SOURCE_ROOT))

from owned_agent_control import __version__  # noqa: E402
import owned_agent_control.controller as controller  # noqa: E402
from owned_agent_control.controller import (  # noqa: E402
    CONTEXT_EVIDENCE_SCHEMA,
    PROGRAM_VERSION,
    ContextGenerationError,
    ControlError,
    CrossStepConsistency,
    EnrichedContextGenerator,
    EntropyDepthAllocator,
    build_parser,
    build_request,
    canonical_json,
    connect,
    create_state_database,
    export_a11oy_context_evidence,
    generate_context,
    add_operator_raw,
    apply_isolation,
    append_audit,
    preflight_supervisor_runtime,
    read_context_trace,
    record_context_trace,
    register_agent,
    require_stabilized_context,
    seal_trust_store,
    sign_request_with_key,
    state_paths,
    verify_audit,
    verify_request,
)


FIXED_TIME = "2026-08-20T12:00:00.000000Z"
INITIAL_TIME = "2000-01-01T00:00:00.000000Z"
FIXED_TRACE_ID = "dca30da9-40a3-4f29-96f8-b98c74bb7d85"
TARGET = "owned-agent:test"


def stable_context_input() -> dict[str, object]:
    invariants = {
        "a11oy_authority": "read_only_projection",
        "enforcement_boundary": "local_windows_supervised_processes_only",
    }
    transitions = [
        ("Observe", "Analyze"),
        ("Analyze", "Decide"),
        ("Decide", "Verify"),
        ("Verify", "Stabilized"),
    ]
    return {
        "challenge_question": 20260820,
        "steps": [
            {
                "context_reweighted": False,
                "from_state": source,
                "invariants": dict(invariants),
                "output_filter_triggered": False,
                "safety_gate_passed": True,
                "to_state": destination,
            }
            for source, destination in transitions
        ],
    }


def target_row() -> dict[str, object]:
    argv = [str(Path(sys.executable).resolve()), "-c", "pass"]
    return {
        "argv_json": canonical_json(argv).decode("utf-8"),
        "control_state": "READY",
        "executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        "target": TARGET,
    }


class PackageContractTests(unittest.TestCase):
    def test_test_dependency_closure_and_workflow_install_are_exact(self) -> None:
        requirements = (PACKAGE_ROOT / "requirements-test.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("--only-binary=:all:", requirements)
        self.assertIn("--require-hashes", requirements)
        pins = dict(
            re.findall(
                r"^([A-Za-z0-9-]+)==([^\\;\s]+)",
                requirements,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(
            pins,
            {
                "attrs": "26.1.0",
                "jsonschema": "4.26.0",
                "jsonschema-specifications": "2025.9.1",
                "referencing": "0.37.0",
                "rpds-py": "2026.6.3",
                "typing-extensions": "4.16.0",
            },
        )
        self.assertIn('python_version < "3.13"', requirements)
        self.assertEqual(
            len(re.findall(r"--hash=sha256:[0-9a-f]{64}", requirements)),
            11,
        )
        workflow = (
            PACKAGE_ROOT.parent / ".github" / "workflows" / "owned-agent-control.yml"
        ).read_text(encoding="utf-8")
        install_suffix = "--no-deps --requirement agent-forge/requirements-test.txt"
        self.assertEqual(workflow.count(install_suffix), 2)
        self.assertNotIn(
            "--disable-pip-version-check --requirement agent-forge/requirements-test.txt",
            workflow,
        )

    def test_versions_are_aligned(self) -> None:
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(__version__, "2.0.0")
        self.assertEqual(PROGRAM_VERSION, __version__)
        self.assertIn('version = "2.0.0"', pyproject)

    def test_schema_copies_are_semantically_identical_and_read_only(self) -> None:
        repository_schema = json.loads(
            (
                PACKAGE_ROOT
                / "schemas"
                / "a11oy-owned-agent-control-projection.schema.json"
            ).read_text(encoding="utf-8")
        )
        package_schema = json.loads(
            (
                SOURCE_ROOT
                / "owned_agent_control"
                / "schemas"
                / "a11oy-owned-agent-control-projection.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(repository_schema, package_schema)
        self.assertEqual(
            repository_schema["properties"]["schema"]["const"],
            "a11oy/owned-agent-control-projection/v1",
        )
        capabilities = repository_schema["$defs"]["capabilities"]["properties"]
        self.assertIs(capabilities["a11oy_process_control"]["const"], False)
        self.assertIs(capabilities["a11oy_read_only_projection"]["const"], True)
        scalar_types = repository_schema["$defs"]["jsonScalar"]["anyOf"]
        self.assertNotIn({"type": "number"}, scalar_types)
        self.assertIn(
            {
                "type": "integer",
                "minimum": -(2**63),
                "maximum": 2**63 - 1,
            },
            scalar_types,
        )
        self.assertIn(
            "integral number forms",
            repository_schema["$defs"]["jsonScalar"]["$comment"],
        )

    def test_context_json_parsing_accepts_only_integral_float_forms(self) -> None:
        with self.assertRaises(ControlError) as strict:
            controller.parse_json_bytes(b'{"value":1.0}')
        self.assertEqual(strict.exception.code, "INVALID_JSON_TYPE")
        parsed = controller.parse_json_bytes(
            b'{"value":1.0}', allow_integral_floats=True
        )
        self.assertIs(type(parsed["value"]), float)
        self.assertEqual(parsed["value"], 1.0)
        with self.assertRaises(ControlError) as fractional:
            controller.parse_json_bytes(
                b'{"value":0.5}', allow_integral_floats=True
            )
        self.assertEqual(fractional.exception.code, "INVALID_JSON_TYPE")

    def test_context_json_parsing_rejects_lossy_integral_float_forms(self) -> None:
        parsed = controller.parse_json_bytes(
            b'{"value":9007199254740992.0}', allow_integral_floats=True
        )
        self.assertIs(type(parsed["value"]), float)
        self.assertEqual(parsed["value"], 9007199254740992.0)
        reparsed = controller.parse_json_bytes(
            canonical_json(parsed), allow_integral_floats=True
        )
        self.assertEqual(reparsed, parsed)

        for literal in (
            "-9223372036854775808.0",
            "9007199254740993.0",
            "9223372036854775807.0",
            "9223372036854775808.0",
        ):
            with self.subTest(literal=literal):
                with self.assertRaises(ControlError) as lossy:
                    controller.parse_json_bytes(
                        f'{{"value":{literal}}}'.encode("ascii"),
                        allow_integral_floats=True,
                    )
                self.assertEqual(lossy.exception.code, "INVALID_JSON_TYPE")

    def test_cli_surface_contains_operator_and_context_commands(self) -> None:
        parser = build_parser()
        choices: set[str] = set()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                choices.update(action.choices)
        self.assertTrue(
            {
                "audit-verify",
                "apply-isolation",
                "context-export",
                "context-generate",
                "context-show",
                "doctor",
                "init",
                "keygen",
                "operator-add",
                "register",
                "register-demo",
                "request-new",
                "request-sign",
                "request-verify",
                "self-test",
                "start",
                "status",
                "trust-seal",
            }.issubset(choices)
        )

    def test_exported_projection_validates_against_full_schema(self) -> None:
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError:
            self.skipTest("jsonschema test dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory)
            paths = state_paths(temporary_path / "state")
            paths.root.mkdir()
            paths.logs.mkdir()
            paths.demo.mkdir()
            create_state_database(paths.database, initialized_at=INITIAL_TIME)
            register_agent(
                paths,
                TARGET,
                [str(Path(sys.executable).resolve()), "-c", "pass"],
                temporary_path,
            )
            context_input = stable_context_input()
            steps = context_input["steps"]
            assert isinstance(steps, list)
            steps[0]["invariants"]["confidence"] = 1.0
            for step in steps[1:]:
                step["invariants"]["confidence"] = 1
            generated = generate_context(paths, TARGET, context_input)
            self.assertIs(
                type(generated["execution_trace"][0]["invariants"]["confidence"]),
                float,
            )
            self.assertNotIn(
                "confidence", generated["consistency"]["conflicting_invariants"]
            )
            projection = export_a11oy_context_evidence(paths, TARGET)
            self.assertIs(
                type(
                    projection["context"]["execution_trace"][0]["invariants"][
                        "confidence"
                    ]
                ),
                float,
            )
            schema = json.loads(
                (
                    PACKAGE_ROOT
                    / "schemas"
                    / "a11oy-owned-agent-control-projection.schema.json"
                ).read_text(encoding="utf-8")
            )
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(
                projection
            )


class ContextAlgorithmTests(unittest.TestCase):
    def test_entropy_allocator_counts_reweighting(self) -> None:
        steps = stable_context_input()["steps"]
        assert isinstance(steps, list)
        steps[1]["context_reweighted"] = True
        self.assertEqual(EntropyDepthAllocator(3).allocate(steps), 1)

    def test_entropy_allocator_fails_closed_over_budget(self) -> None:
        steps = stable_context_input()["steps"]
        assert isinstance(steps, list)
        steps[0]["context_reweighted"] = True
        steps[1]["context_reweighted"] = True
        with self.assertRaises(ContextGenerationError) as caught:
            EntropyDepthAllocator(1).allocate(steps)
        self.assertEqual(caught.exception.code, "ENTROPY_BUDGET_EXCEEDED")

    def test_consistency_scores_repeated_invariants(self) -> None:
        steps = stable_context_input()["steps"]
        assert isinstance(steps, list)
        result = CrossStepConsistency(950_000).evaluate(steps)
        self.assertGreater(result["comparisons"], 0)
        self.assertEqual(result["score_ppm"], 1_000_000)
        self.assertEqual(result["conflicting_invariants"], [])
        self.assertIs(result["threshold_met"], True)

    def test_consistency_empty_comparisons_do_not_pass(self) -> None:
        step = stable_context_input()["steps"][0]
        result = CrossStepConsistency(950_000).evaluate([step])
        self.assertEqual(result["comparisons"], 0)
        self.assertEqual(result["score_ppm"], 0)
        self.assertIs(result["threshold_met"], False)

    def test_generator_is_deterministic_with_fixed_identity_and_time(self) -> None:
        first = EnrichedContextGenerator(target_row()).generate(
            stable_context_input(), created_at=FIXED_TIME, trace_id=FIXED_TRACE_ID
        )
        second = EnrichedContextGenerator(target_row()).generate(
            stable_context_input(), created_at=FIXED_TIME, trace_id=FIXED_TRACE_ID
        )
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], CONTEXT_EVIDENCE_SCHEMA)
        self.assertEqual(first["convergence"], "Stabilized")
        self.assertEqual(len(first["evidence_sha256"]), 64)

    def test_generator_rejects_invalid_created_at_before_hashing(self) -> None:
        with self.assertRaises(ContextGenerationError) as caught:
            EnrichedContextGenerator(target_row()).generate(
                stable_context_input(), created_at="not-a-timestamp", trace_id=FIXED_TRACE_ID
            )
        self.assertEqual(caught.exception.code, "INVALID_CONTEXT_TIMESTAMP")

    def test_generator_canonicalizes_valid_created_at_before_hashing(self) -> None:
        evidence = EnrichedContextGenerator(target_row()).generate(
            stable_context_input(), created_at="2026-08-20T12:00:00Z", trace_id=FIXED_TRACE_ID
        )
        self.assertEqual(evidence["created_at"], FIXED_TIME)

    def test_generator_flags_conflicting_invariants(self) -> None:
        context_input = stable_context_input()
        context_input["steps"][1]["invariants"]["a11oy_authority"] = "process_control"
        evidence = EnrichedContextGenerator(target_row()).generate(
            context_input, created_at=FIXED_TIME, trace_id=FIXED_TRACE_ID
        )
        self.assertEqual(evidence["convergence"], "Flagged_For_Review")
        self.assertIn(
            "a11oy_authority", evidence["consistency"]["conflicting_invariants"]
        )


class AtomicStateTests(unittest.TestCase):
    def test_failed_schema_build_is_retryable_and_instance_identity_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "control.sqlite3"
            with mock.patch.object(
                controller, "STATE_SCHEMA_SQL", "CREATE TABLE broken("
            ):
                with self.assertRaises(sqlite3.Error):
                    create_state_database(database, initialized_at=INITIAL_TIME)
            self.assertFalse(database.exists())
            self.assertFalse(Path(f"{database}-journal").exists())
            created = create_state_database(database, initialized_at=INITIAL_TIME)
            with closing(sqlite3.connect(database)) as connection:
                connection.row_factory = sqlite3.Row
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE metadata SET value=? WHERE key='controller_instance_id'",
                        (str(uuid.uuid4()),),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "UPDATE metadata SET key='renamed' WHERE key='controller_instance_id'"
                    )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='controller_instance_id'"
                    ).fetchone()["value"],
                    created["controller_instance_id"],
                )
                self.assertEqual(verify_audit(connection)["events_verified"], 1)

    def test_preflight_empty_stderr_is_a_controlled_failure(self) -> None:
        completed = SimpleNamespace(returncode=1, stderr="")
        with mock.patch.object(controller.subprocess, "run", return_value=completed):
            with self.assertRaises(ControlError) as caught:
                preflight_supervisor_runtime()
        self.assertEqual(caught.exception.code, "SUPERVISOR_RUNTIME_PREFLIGHT_FAILED")
        self.assertIn("isolated interpreter exited 1", caught.exception.message)


class AuthorizationBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.keys = [Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()]
        self.primary = self._new_root("primary")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _new_root(self, name: str):
        paths = state_paths(self.base / name)
        paths.root.mkdir()
        paths.logs.mkdir()
        paths.demo.mkdir()
        create_state_database(paths.database, initialized_at=INITIAL_TIME)
        register_agent(
            paths,
            TARGET,
            [str(Path(sys.executable).resolve()), "-c", "pass"],
            self.base,
        )
        add_operator_raw(paths, "alice", self.keys[0].public_key())
        add_operator_raw(paths, "bob", self.keys[1].public_key())
        seal_trust_store(paths)
        return paths

    def _signed_request(self) -> dict[str, object]:
        envelope = build_request(self.primary, TARGET, 300)
        envelope = sign_request_with_key(self.primary, envelope, "alice", self.keys[0])
        return sign_request_with_key(self.primary, envelope, "bob", self.keys[1])

    def test_signed_request_cannot_cross_controller_roots(self) -> None:
        other = self._new_root("other")
        with self.assertRaises(ControlError) as caught:
            verify_request(other, self._signed_request())
        self.assertEqual(caught.exception.code, "CONTROLLER_INSTANCE_MISMATCH")

    def test_tampered_target_binding_is_rejected(self) -> None:
        envelope = self._signed_request()
        envelope["target_binding"]["argv_sha256"] = "f" * 64
        with self.assertRaises(ControlError) as caught:
            verify_request(self.primary, envelope)
        self.assertEqual(caught.exception.code, "TARGET_BINDING_MISMATCH")

    def test_full_signed_envelope_is_persisted_canonically(self) -> None:
        envelope = self._signed_request()
        with (
            mock.patch.object(controller.os, "name", "nt"),
            mock.patch.object(controller, "require_no_reparse_ancestors"),
            mock.patch.object(controller, "_is_reparse_point", return_value=False),
        ):
            result = apply_isolation(self.primary, envelope)
        self.assertEqual(result["operation_status"], "VERIFIED_ISOLATED")
        with connect(self.primary) as connection:
            row = connection.execute(
                "SELECT envelope_json, status, result_json FROM requests WHERE request_id=?",
                (envelope["request_id"],),
            ).fetchone()
            self.assertEqual(row["status"], "APPLIED")
            self.assertEqual(
                row["envelope_json"], canonical_json(envelope).decode("utf-8")
            )
            stored = json.loads(row["envelope_json"])
            self.assertEqual(
                sorted(stored["authorization"]["signatures"]), ["alice", "bob"]
            )
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )
            original_result = json.loads(row["result_json"])
            tampered_result = copy.deepcopy(original_result)
            tampered_result["enforcement"]["provider_credentials_revoked"] = True
            tampered_result["truth_boundary"] = "remote_provider_control_confirmed"
            tampered_json = canonical_json(tampered_result).decode("utf-8")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE requests SET result_json=? WHERE request_id=?",
                    (tampered_json, envelope["request_id"]),
                )
            trigger_sql = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='trigger' AND name='requests_applied_no_update'
                """
            ).fetchone()["sql"]
            connection.execute("DROP TRIGGER requests_applied_no_update")
            connection.execute(
                "UPDATE requests SET result_json=? WHERE request_id=?",
                (tampered_json, envelope["request_id"]),
            )
            connection.execute(trigger_sql)
            with self.assertRaisesRegex(
                controller.IntegrityFailure, "result does not match its audit event"
            ):
                verify_audit(connection)

    def test_unclaimed_isolation_never_claims_supervisor_exit_or_job_cleanup(
        self,
    ) -> None:
        generated = generate_context(self.primary, TARGET, stable_context_input())
        run_id = str(uuid.uuid4())
        now = controller.format_time(controller.utc_now())
        with connect(self.primary) as connection:
            with controller.immediate_transaction(connection):
                target = connection.execute(
                    "SELECT * FROM targets WHERE target=?", (TARGET,)
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, target, context_trace_id, state, job_name,
                        supervisor_token_hash, log_path, created_at
                    ) VALUES (?, ?, ?, 'STARTING', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        TARGET,
                        generated["trace_id"],
                        f"UNNAMED:{run_id}",
                        "a" * 64,
                        str(self.primary.logs / f"{run_id}.log"),
                        now,
                    ),
                )
                append_audit(
                    connection,
                    "START_RESERVED",
                    TARGET,
                    "test",
                    {
                        "context_trace_id": generated["trace_id"],
                        "run_id": run_id,
                        "shell": False,
                        "target_binding": controller.target_binding_from_row(target),
                    },
                )
        envelope = self._signed_request()
        with (
            mock.patch.object(controller.os, "name", "nt"),
            mock.patch.object(controller, "require_no_reparse_ancestors"),
            mock.patch.object(controller, "_is_reparse_point", return_value=False),
            mock.patch.object(
                controller,
                "query_process_status",
                side_effect=AssertionError("unclaimed identity must not be queried"),
            ),
        ):
            result = apply_isolation(self.primary, envelope, timeout_seconds=0.1)
        self.assertEqual(result["operation_status"], "VERIFIED_ISOLATED")
        self.assertEqual(
            result["enforcement"]["local_process_tree_absence_basis"],
            "no_live_supervised_process_tree",
        )
        self.assertIs(
            result["enforcement"]["local_process_tree_termination_performed"], False
        )
        with connect(self.primary) as connection:
            run = connection.execute(
                """
                SELECT state, supervisor_pid, supervisor_created_filetime,
                    supervisor_token_hash FROM runs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
            self.assertEqual(run["state"], "ISOLATED")
            self.assertIsNone(run["supervisor_pid"])
            self.assertIsNone(run["supervisor_created_filetime"])
            self.assertIsNone(run["supervisor_token_hash"])
            false_exit_claims = connection.execute(
                """
                SELECT COUNT(*) AS count FROM audit_events
                WHERE event_type='SUPERVISOR_EXIT_RECONCILED'
                """
            ).fetchone()["count"]
            self.assertEqual(int(false_exit_claims), 0)
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_applied_no_live_isolation_does_not_rewrite_failed_run(self) -> None:
        generated = generate_context(self.primary, TARGET, stable_context_input())
        run_id = str(uuid.uuid4())
        now = controller.format_time(controller.utc_now())
        with connect(self.primary) as connection:
            with controller.immediate_transaction(connection):
                target = connection.execute(
                    "SELECT * FROM targets WHERE target=?", (TARGET,)
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, target, context_trace_id, state, job_name,
                        log_path, error_code, error_message, ended_at, created_at
                    ) VALUES (?, ?, ?, 'FAILED', ?, ?,
                        'SYNTHETIC_START_FAILURE', 'synthetic pre-claim failure', ?, ?)
                    """,
                    (
                        run_id,
                        TARGET,
                        generated["trace_id"],
                        f"UNNAMED:{run_id}",
                        str(self.primary.logs / f"{run_id}.log"),
                        now,
                        now,
                    ),
                )
                append_audit(
                    connection,
                    "START_RESERVED",
                    TARGET,
                    "test",
                    {
                        "context_trace_id": generated["trace_id"],
                        "run_id": run_id,
                        "shell": False,
                        "target_binding": controller.target_binding_from_row(target),
                    },
                )
                append_audit(
                    connection,
                    "START_FAILED",
                    TARGET,
                    "test",
                    {"error_code": "SYNTHETIC_START_FAILURE", "run_id": run_id},
                )
        envelope = self._signed_request()
        with (
            mock.patch.object(controller.os, "name", "nt"),
            mock.patch.object(controller, "require_no_reparse_ancestors"),
            mock.patch.object(controller, "_is_reparse_point", return_value=False),
            mock.patch.object(
                controller,
                "query_process_status",
                side_effect=AssertionError("unclaimed identity must not be queried"),
            ),
        ):
            applied = apply_isolation(self.primary, envelope, timeout_seconds=0.1)
            reconciled = controller.reconcile_stale_runs(self.primary, TARGET)
        self.assertEqual(applied["operation_status"], "VERIFIED_ISOLATED")
        self.assertIsNone(applied["run_id"])
        self.assertEqual(
            reconciled[0]["operation_status"],
            "VERIFIED_ISOLATED_BEFORE_LAUNCH_GATE",
        )
        with connect(self.primary) as connection:
            run = connection.execute(
                "SELECT state, error_code FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            self.assertEqual(run["state"], "FAILED")
            self.assertEqual(run["error_code"], "SYNTHETIC_START_FAILURE")
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )


class PersistentEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary.name)
        self.paths = state_paths(temporary_path / "state")
        self.paths.root.mkdir()
        self.paths.logs.mkdir()
        self.paths.demo.mkdir()
        create_state_database(self.paths.database, initialized_at=INITIAL_TIME)
        register_agent(
            self.paths,
            TARGET,
            [str(Path(sys.executable).resolve()), "-c", "pass"],
            temporary_path,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_state_database_uses_strict_tables_and_immutable_target_binding(
        self,
    ) -> None:
        with connect(self.paths) as connection:
            strict = connection.execute(
                "SELECT strict FROM pragma_table_list WHERE name='targets'"
            ).fetchone()
            self.assertEqual(int(strict["strict"]), 1)
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE targets SET executable_sha256=? WHERE target=?",
                    ("f" * 64, TARGET),
                )

    def test_connection_context_closes_database_handle(self) -> None:
        connection = connect(self.paths)
        with connection as entered:
            self.assertIs(entered, connection)
            entered.execute("SELECT 1").fetchone()
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_run_context_constraints_enforce_single_use_and_target_binding(
        self,
    ) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        now = controller.format_time(controller.utc_now())
        with connect(self.paths) as connection:
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, target, context_trace_id, state, job_name, log_path, created_at
                ) VALUES (?, ?, ?, 'FAILED', ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    TARGET,
                    generated["trace_id"],
                    f"UNNAMED:{uuid.uuid4()}",
                    str(self.paths.logs / "first.log"),
                    now,
                ),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, target, context_trace_id, state, job_name, log_path, created_at
                    ) VALUES (?, ?, ?, 'FAILED', ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        TARGET,
                        generated["trace_id"],
                        f"UNNAMED:{uuid.uuid4()}",
                        str(self.paths.logs / "duplicate.log"),
                        now,
                    ),
                )
            connection.execute("ROLLBACK")
        other_target = "owned-agent:other"
        register_agent(
            self.paths,
            other_target,
            [str(Path(sys.executable).resolve()), "-c", "pass"],
            Path(self.temporary.name),
        )
        with connect(self.paths) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, target, context_trace_id, state, job_name, log_path, created_at
                    ) VALUES (?, ?, ?, 'FAILED', ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        other_target,
                        generated["trace_id"],
                        f"UNNAMED:{uuid.uuid4()}",
                        str(self.paths.logs / "cross-target.log"),
                        now,
                    ),
                )

    def test_claimed_supervisor_identity_is_monotonic_and_audit_bound(self) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        run_id = str(uuid.uuid4())
        now = controller.format_time(controller.utc_now())
        supervisor_pid = 4242
        supervisor_created_filetime = 133_700_000_000_000_000
        child_pid = 4343
        with connect(self.paths) as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                "SELECT * FROM targets WHERE target=?", (TARGET,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, target, context_trace_id, state, job_name,
                    supervisor_pid, supervisor_created_filetime, child_pid,
                    log_path, created_at
                ) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    TARGET,
                    generated["trace_id"],
                    f"UNNAMED:{run_id}",
                    supervisor_pid,
                    supervisor_created_filetime,
                    child_pid,
                    str(self.paths.logs / f"{run_id}.log"),
                    now,
                ),
            )
            append_audit(
                connection,
                "START_RESERVED",
                TARGET,
                "test",
                {
                    "context_trace_id": generated["trace_id"],
                    "run_id": run_id,
                    "shell": False,
                    "target_binding": controller.target_binding_from_row(target),
                },
            )
            append_audit(
                connection,
                "SUPERVISOR_CLAIMED",
                TARGET,
                "test",
                {
                    "run_id": run_id,
                    "supervisor_pid": supervisor_pid,
                    "supervisor_created_filetime": supervisor_created_filetime,
                },
            )
            append_audit(
                connection,
                "PROCESS_TREE_STARTED",
                TARGET,
                "test",
                {"child_pid": child_pid, "run_id": run_id},
            )
            connection.execute("COMMIT")
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    UPDATE runs
                    SET supervisor_pid=NULL, supervisor_created_filetime=NULL
                    WHERE run_id=?
                    """,
                    (run_id,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE runs SET supervisor_pid=? WHERE run_id=?",
                    (supervisor_pid + 1, run_id),
                )

            trigger_sql = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='trigger' AND name='runs_supervisor_identity_monotonic'
                """
            ).fetchone()["sql"]
            connection.execute("DROP TRIGGER runs_supervisor_identity_monotonic")
            connection.execute(
                """
                UPDATE runs
                SET supervisor_pid=NULL, supervisor_created_filetime=NULL
                WHERE run_id=?
                """,
                (run_id,),
            )
            connection.execute(trigger_sql)
            with self.assertRaisesRegex(
                controller.IntegrityFailure, "supervisor identity is incomplete"
            ):
                verify_audit(connection)

    def test_parent_never_persists_unaudited_bootstrap_supervisor_identity(
        self,
    ) -> None:
        keys = [Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()]
        add_operator_raw(self.paths, "alice", keys[0].public_key())
        add_operator_raw(self.paths, "bob", keys[1].public_key())
        seal_trust_store(self.paths)
        generate_context(self.paths, TARGET, stable_context_input())
        with (
            mock.patch.object(controller.os, "name", "nt"),
            mock.patch.object(controller, "ensure_state_root"),
            mock.patch.object(controller, "_is_reparse_point", return_value=False),
            mock.patch.object(controller, "preflight_supervisor_runtime"),
            mock.patch.object(controller, "reconcile_stale_runs"),
            mock.patch.object(
                controller, "spawn_run_supervisor", return_value=(4242, 0)
            ),
            mock.patch.object(controller.time, "monotonic", side_effect=(0.0, 1.0)),
        ):
            with self.assertRaises(ControlError) as caught:
                controller.start_agent(self.paths, TARGET, timeout_seconds=0.1)
        self.assertEqual(caught.exception.code, "START_UNCONFIRMED")
        with connect(self.paths) as connection:
            row = connection.execute(
                "SELECT state, supervisor_pid, supervisor_created_filetime FROM runs"
            ).fetchone()
            self.assertEqual(row["state"], "STARTING")
            self.assertIsNone(row["supervisor_pid"])
            self.assertIsNone(row["supervisor_created_filetime"])
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_unclaimed_reservation_times_out_without_process_or_job_claims(
        self,
    ) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        run_id = str(uuid.uuid4())
        with connect(self.paths) as connection:
            with controller.immediate_transaction(connection):
                target = connection.execute(
                    "SELECT * FROM targets WHERE target=?", (TARGET,)
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, target, context_trace_id, state, job_name,
                        supervisor_token_hash, log_path, created_at
                    ) VALUES (?, ?, ?, 'STARTING', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        TARGET,
                        generated["trace_id"],
                        f"UNNAMED:{run_id}",
                        "a" * 64,
                        str(self.paths.logs / f"{run_id}.log"),
                        INITIAL_TIME,
                    ),
                )
                append_audit(
                    connection,
                    "START_RESERVED",
                    TARGET,
                    "test",
                    {
                        "context_trace_id": generated["trace_id"],
                        "run_id": run_id,
                        "shell": False,
                        "target_binding": controller.target_binding_from_row(target),
                    },
                )
        with (
            mock.patch.object(controller.os, "name", "nt"),
            mock.patch.object(controller, "require_no_reparse_ancestors"),
            mock.patch.object(controller, "_is_reparse_point", return_value=False),
            mock.patch.object(
                controller,
                "query_process_status",
                side_effect=AssertionError("unclaimed identity must not be queried"),
            ),
        ):
            reconciled = controller.reconcile_stale_runs(self.paths, TARGET)
        self.assertEqual(reconciled[0]["operation_status"], "SUPERVISOR_CLAIM_TIMEOUT")
        with connect(self.paths) as connection:
            row = connection.execute(
                """
                SELECT state, error_code, error_message, supervisor_token_hash
                FROM runs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
            self.assertEqual(row["state"], "FAILED")
            self.assertEqual(row["error_code"], "SUPERVISOR_CLAIM_TIMEOUT")
            self.assertIn("launch gate was never passed", row["error_message"])
            self.assertIsNone(row["supervisor_token_hash"])
            latest = connection.execute(
                "SELECT event_type, payload_json FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            self.assertEqual(latest["event_type"], "START_FAILED")
            self.assertEqual(
                json.loads(latest["payload_json"])["error_code"],
                "SUPERVISOR_CLAIM_TIMEOUT",
            )
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_audit_time_clamps_clock_rollback_before_commit(self) -> None:
        rolled_back = datetime(1999, 1, 1, tzinfo=timezone.utc)
        with mock.patch.object(controller, "utc_now", return_value=rolled_back):
            register_agent(
                self.paths,
                "owned-agent:clock-rollback",
                [str(Path(sys.executable).resolve()), "-c", "pass"],
                Path(self.temporary.name),
            )
        with connect(self.paths) as connection:
            timestamps = [
                row["timestamp"]
                for row in connection.execute(
                    "SELECT timestamp FROM audit_events ORDER BY sequence"
                )
            ]
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_context_round_trip_and_stabilized_gate(self) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        loaded = read_context_trace(self.paths, target=TARGET)
        self.assertEqual(loaded, generated)
        self.assertEqual(require_stabilized_context(self.paths, TARGET), generated)

    def test_fractional_context_is_rejected_without_poisoning_evidence(
        self,
    ) -> None:
        for invalid_value in (0.5, -9223372036854775808.0):
            with self.subTest(invalid_value=invalid_value):
                context_input = stable_context_input()
                context_input["steps"][0]["invariants"]["confidence"] = invalid_value
                with self.assertRaises(ContextGenerationError) as caught:
                    generate_context(self.paths, TARGET, context_input)
                self.assertEqual(caught.exception.code, "INVALID_CONTEXT_INPUT")
        with connect(self.paths) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM context_traces").fetchone()[0],
                0,
            )
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )
        generated = generate_context(self.paths, TARGET, stable_context_input())
        self.assertEqual(read_context_trace(self.paths, target=TARGET), generated)

    def test_invalid_context_timestamp_is_rejected_before_trace_append(self) -> None:
        with connect(self.paths) as connection:
            target = connection.execute(
                "SELECT * FROM targets WHERE target=?", (TARGET,)
            ).fetchone()
            before = verify_audit(connection)
        evidence = EnrichedContextGenerator(target).generate(
            stable_context_input(),
            created_at=FIXED_TIME,
            trace_id=FIXED_TRACE_ID,
        )
        evidence["created_at"] = "not-rfc3339"
        evidence["evidence_sha256"] = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "evidence_sha256"
                }
            )
        ).hexdigest()
        with self.assertRaises(ContextGenerationError) as caught:
            record_context_trace(self.paths, evidence)
        self.assertEqual(caught.exception.code, "INVALID_CONTEXT_EVIDENCE")
        with connect(self.paths) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM context_traces").fetchone()[0],
                0,
            )
            self.assertEqual(verify_audit(connection), before)

    def test_register_rejects_relative_working_directory_before_normalization(
        self,
    ) -> None:
        with self.assertRaises(ControlError) as caught:
            register_agent(
                self.paths,
                "owned-agent:relative-cwd",
                [str(Path(sys.executable).resolve()), "-c", "pass"],
                Path("."),
            )
        self.assertEqual(caught.exception.code, "INVALID_WORKING_DIRECTORY")
        with connect(self.paths) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT target FROM targets WHERE target=?",
                    ("owned-agent:relative-cwd",),
                ).fetchone()
            )
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_context_trace_is_append_only(self) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        with connect(self.paths) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM context_traces WHERE trace_id=?",
                    (generated["trace_id"],),
                )

    def test_mutated_context_evidence_is_rejected(self) -> None:
        evidence = EnrichedContextGenerator(target_row()).generate(
            stable_context_input()
        )
        evidence["challenge_question"] = 7
        with self.assertRaises(ContextGenerationError) as caught:
            record_context_trace(self.paths, evidence)
        self.assertEqual(caught.exception.code, "CONTEXT_HASH_MISMATCH")

    def test_multiple_context_traces_are_preserved_and_latest_is_selected(self) -> None:
        first = generate_context(self.paths, TARGET, stable_context_input())
        second = generate_context(self.paths, TARGET, stable_context_input())
        self.assertNotEqual(first["trace_id"], second["trace_id"])
        self.assertEqual(
            read_context_trace(self.paths, trace_id=first["trace_id"]), first
        )
        self.assertEqual(read_context_trace(self.paths, target=TARGET), second)

    def test_projection_is_read_only_hash_bound_and_audit_verified(self) -> None:
        generate_context(self.paths, TARGET, stable_context_input())
        projection = export_a11oy_context_evidence(self.paths, TARGET)
        self.assertEqual(
            projection["schema"], "a11oy/owned-agent-control-projection/v1"
        )
        self.assertIs(projection["capabilities"]["a11oy_process_control"], False)
        self.assertIs(projection["capabilities"]["a11oy_read_only_projection"], True)
        self.assertIs(projection["truth_boundary"]["remote_effects"], False)
        self.assertIs(projection["truth_boundary"]["semantic_safety_evaluation"], False)
        unsigned = {
            key: value
            for key, value in projection.items()
            if key != "projection_sha256"
        }
        self.assertEqual(
            projection["projection_sha256"],
            hashlib.sha256(canonical_json(unsigned)).hexdigest(),
        )
        with connect(self.paths) as connection:
            self.assertEqual(
                verify_audit(connection)["integrity"], "VERIFIED_LOCAL_HASH_CHAIN"
            )

    def test_stabilized_context_is_single_use_for_start_reservations(self) -> None:
        generated = generate_context(self.paths, TARGET, stable_context_input())
        run_id = str(uuid.uuid4())
        now = controller.format_time(controller.utc_now())
        with connect(self.paths) as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                "SELECT * FROM targets WHERE target=?", (TARGET,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, target, context_trace_id, state, job_name, log_path, created_at,
                    error_code, error_message, ended_at
                ) VALUES (?, ?, ?, 'FAILED', ?, ?, ?, 'TEST_FAILURE', 'synthetic', ?)
                """,
                (
                    run_id,
                    TARGET,
                    generated["trace_id"],
                    f"UNNAMED:{run_id}",
                    str(self.paths.logs / f"{run_id}.log"),
                    now,
                    now,
                ),
            )
            append_audit(
                connection,
                "START_RESERVED",
                TARGET,
                "test",
                {
                    "context_trace_id": generated["trace_id"],
                    "run_id": run_id,
                    "shell": False,
                    "target_binding": controller.target_binding_from_row(target),
                },
            )
            append_audit(
                connection,
                "START_FAILED",
                TARGET,
                "test",
                {"error_code": "TEST_FAILURE", "run_id": run_id},
            )
            connection.execute("COMMIT")
        with self.assertRaises(ControlError) as caught:
            require_stabilized_context(self.paths, TARGET)
        self.assertEqual(caught.exception.code, "CONTEXT_ALREADY_CONSUMED")

    def test_latest_context_uses_insertion_order_not_wall_clock(self) -> None:
        with connect(self.paths) as connection:
            target = dict(
                connection.execute(
                    "SELECT * FROM targets WHERE target=?", (TARGET,)
                ).fetchone()
            )
        first = EnrichedContextGenerator(target).generate(
            stable_context_input(),
            created_at="2099-01-01T00:00:00.000000Z",
            trace_id="11111111-1111-4111-8111-111111111111",
        )
        second = EnrichedContextGenerator(target).generate(
            stable_context_input(),
            created_at="2001-01-01T00:00:00.000000Z",
            trace_id="22222222-2222-4222-8222-222222222222",
        )
        record_context_trace(self.paths, first)
        record_context_trace(self.paths, second)
        self.assertEqual(read_context_trace(self.paths, target=TARGET), second)


if __name__ == "__main__":
    unittest.main()
