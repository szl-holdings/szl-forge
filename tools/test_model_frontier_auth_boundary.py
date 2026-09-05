#!/usr/bin/env python3
"""Offline workflow structure tests: credentials never enter pull-request jobs."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/model-kernel-frontier.yml'
GUARD = (
    "github.repository == 'szl-holdings/szl-forge' && "
    "github.ref == 'refs/heads/main' && "
    "(github.event_name == 'push' || github.event_name == 'workflow_dispatch')"
)


class ModelFrontierAuthBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text(encoding='utf-8')
        # BaseLoader preserves GitHub's YAML `on` key instead of YAML 1.1 booleans.
        self.doc = yaml.load(self.text, Loader=yaml.BaseLoader)
        self.jobs = self.doc['jobs']
        self.local = self.jobs['verify']
        self.live = self.jobs['verify-hub-evidence']

    def test_pull_request_events_are_not_privileged(self):
        self.assertIn('pull_request', self.doc['on'])
        self.assertNotIn('pull_request_target', self.doc['on'])
        self.assertEqual(self.doc['permissions'], {'contents': 'read'})
        for name, job in self.jobs.items():
            if name == 'verify-hub-evidence':
                continue
            self.assertNotIn('secrets.', yaml.dump(job))
            self.assertNotIn('id-token', yaml.dump(job))

    def test_authenticated_job_requires_exact_trusted_events_and_both_source_gates(self):
        self.assertEqual(' '.join(self.live['if'].split()), GUARD)
        self.assertEqual(set(self.live['needs']), {'verify', 'verify-receiptagent-v3-windows'})
        self.assertEqual(self.live['permissions'], {'contents': 'read'})
        self.assertEqual(self.live['timeout-minutes'], '15')

    def test_authenticated_checkout_uses_event_commit_without_persisted_git_credential(self):
        checkout = next(step for step in self.live['steps'] if str(step.get('uses', '')).startswith('actions/checkout@'))
        self.assertEqual(checkout['with']['ref'], '${{ github.sha }}')
        self.assertEqual(checkout['with']['persist-credentials'], 'false')
        self.assertNotIn('pull_request', str(checkout))

    def test_original_cryptographic_and_portfolio_tests_are_retained(self):
        runs = '\n'.join(step.get('run', '') for step in self.local['steps'])
        for command in (
            'python tools/test_verify_model_portfolio.py',
            'python tools/test_publish_model_source_bindings.py',
            'python tools/test_model_binding_workflow.py',
            'python tools/verify_model_portfolio.py --offline',
            'frontier/qwen35-receiptagent-v2/evidence_chain.py verify',
            'tools/test_publish_receiptagent_v3.py',
            'test_model_frontier_auth_boundary.py',
        ):
            self.assertIn(command, runs)
        self.assertIn('Build the stable Kernel runtime image', [s.get('name') for s in self.local['steps']])
        self.assertIn('Verify isolated failure evidence round trip', [s.get('name') for s in self.local['steps']])

    def test_no_live_binding_or_live_portfolio_command_in_the_pr_source_job(self):
        runs = '\n'.join(step.get('run', '') for step in self.local['steps'])
        self.assertNotRegex(runs, r'(?m)^\s*python tools/publish_model_source_bindings\.py(?:\s|$)')
        self.assertNotRegex(runs, r'verify_model_portfolio\.py[^\n]*--live')
        self.assertNotIn('HF_TOKEN', runs)
        self.assertIn('--report reports/model-portfolio-offline.json', runs)

    def test_all_remote_receipt_and_portfolio_validations_remain_required(self):
        runs = '\n'.join(step.get('run', '') for step in self.live['steps'])
        self.assertIn('python tools/publish_model_source_bindings.py', runs)
        self.assertIn('--source-revision "${GITHUB_SHA}"', runs)
        self.assertIn('python tools/verify_model_portfolio.py --live', runs)
        self.assertNotIn('--offline', runs)
        self.assertNotIn('continue-on-error', yaml.dump(self.live))
        self.assertNotIn('|| true', runs)
        self.assertNotIn('|| :', runs)

    def test_no_publication_gate_change_or_model_execution(self):
        runs = '\n'.join(step.get('run', '') for step in self.live['steps'])
        self.assertNotRegex(runs, r'(^|\s)--publish(?:\s|$)')
        for prohibited in ('--allow-create', '--clear-space-volumes', 'update_repo_settings',
                           'request_access', 'accept_access', 'train_candidate.py',
                           'verify_governed_live.py', 'run_governed_live_verifier.py'):
            self.assertNotIn(prohibited, runs)

    def test_credentials_are_step_scoped_and_acquired_after_offline_selector_tests(self):
        self.assertNotIn('env', self.live)
        steps = self.live['steps']
        credential_steps = [s for s in steps if 'secrets.' in yaml.dump(s)]
        self.assertEqual(len(credential_steps), 1)
        credential = credential_steps[0]
        self.assertIn('python tools/acquire_hf_publisher_token.py', credential['run'])
        self.assertIn('--target-repo SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent', credential['run'])
        self.assertIn('--target-type model', credential['run'])
        self.assertIn('--github-env "${GITHUB_ENV}"', credential['run'])
        earlier_runs = '\n'.join(s.get('run', '') for s in steps[:steps.index(credential)])
        self.assertIn('test_acquire_hf_publisher_token.py', earlier_runs)
        self.assertIn('test_hf_org_token1_fallback.py', earlier_runs)
        self.assertEqual(set(credential['env']), {
            'HF_ORG_TOKEN_CANDIDATE', 'HF_ORG_TOKEN1_CANDIDATE',
            'HF_WRITE_TOKEN_CANDIDATE', 'HF_TOKEN_CANDIDATE',
            'HUGGINGFACE_TOKEN_CANDIDATE', 'HUGGING_FACE_HUB_TOKEN_CANDIDATE',
        })

    def test_all_actions_are_immutable_and_reports_are_retained_on_failure(self):
        for job in self.jobs.values():
            for step in job['steps']:
                if 'uses' in step:
                    self.assertRegex(step['uses'].split(' #')[0], r'^[^@]+@[0-9a-f]{40}$')
        artifact = next(s for s in self.live['steps'] if str(s.get('uses', '')).startswith('actions/upload-artifact@'))
        self.assertEqual(artifact['if'], 'always()')
        self.assertEqual(artifact['with']['if-no-files-found'], 'error')
        for path in ('reports/hf-model-verifier-credential.json',
                     'reports/model-source-bindings-dry-run.json',
                     'reports/model-portfolio-live.json'):
            self.assertIn(path, artifact['with']['path'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
