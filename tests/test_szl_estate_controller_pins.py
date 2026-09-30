"""Offline controller binding regressions; no live release certification.

Expected identities are transcribed from supplied review evidence E1.
Synthetic bytes below do not represent the reviewed Git blobs.
"""
import base64
import hashlib
import unittest
from unittest.mock import patch

from tools import szl_estate_operator as op

SOURCE = '0f389189edf9930068f1dfbe9cdda851ca197dda'
REVIEWED_PINS = {
    '.github/workflows/estate-release-train.yml': 'fd4611973b1b271d9b8e2bec5d4e0e102b9d1792',
    '.github/workflows/hf-sync.yml': 'e3913a40e9c4884eb60707a0058504ada9e142cc',
    'scripts/estate_repair_dispatch.py': '81f0e83729390ee66b633a8d45e5f1139374ba41',
    'config/estate-release-train.v1.json': '84e998eca4f2f4b6cabc992f12a1c466875be848',
    'scripts/estate_child_completion.py': '077524c306189dd0315e07c492a49081dba17399',
    '.github/workflows/repair-cloudflare-product-edge.yml': 'ad703a8be9601913601d1e53ff070bac87d861a6',
}


class OfflineGH:
    def __init__(self):
        self.calls = []
        self.controls = []
        self.bodies = {path: ('synthetic control: ' + path + '\n').encode()
                       for path in REVIEWED_PINS}
        self.pins = {
            path: hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body,
                               usedforsecurity=False).hexdigest()
            for path, body in self.bodies.items()
        }
        self.bad_path = None
        self.bad_field = None
        self.active_writer = None
        self.rate_limited = False
        self.move_on_recheck = False
        self.branch_reads = 0

    def request(self, endpoint, payload=None, binary=False):
        self.calls.append((endpoint, payload))
        if payload is not None or binary:
            raise AssertionError('offline preflight must only request JSON reads')
        if endpoint == f'{op.PREFIX}/branches/main':
            self.branch_reads += 1
            moved = self.move_on_recheck and self.branch_reads > 1
            return {'name': 'main', 'protected': True,
                    'commit': {'sha': 'b' * 40 if moved else SOURCE}}
        for path, body in self.bodies.items():
            if endpoint == f'{op.PREFIX}/contents/{path}?ref={SOURCE}':
                self.controls.append(path)
                item = {'type': 'file', 'path': path, 'sha': self.pins[path],
                        'encoding': 'base64',
                        'content': base64.b64encode(body).decode('ascii')}
                if path == self.bad_path:
                    if self.bad_field == 'sha':
                        item['sha'] = '0' * 40
                    else:
                        item['content'] = base64.b64encode(body + b'tampered').decode('ascii')
                return item
        if endpoint == f'{op.PREFIX}/actions/workflows/estate-release-train.yml':
            return {'path': op.WORKFLOW_PATH, 'state': 'active'}
        if endpoint in {f'{op.PREFIX}/actions/runs?branch=main&status={state}&per_page=100'
                        for state in op.ACTIVE_STATES}:
            if self.rate_limited:
                raise op.OperatorError('synthetic rate limit')
            runs = [{'path': self.active_writer}] if self.active_writer else []
            return {'total_count': len(runs), 'workflow_runs': runs}
        raise AssertionError('unexpected endpoint: ' + endpoint)


class ControllerPinsTests(unittest.TestCase):
    def test_exact_six_reviewed_identities(self):
        self.assertEqual(op.REPO, 'szl-holdings/a11oy')
        self.assertEqual(op.PINS, REVIEWED_PINS)

    def test_all_six_controls_are_source_bound_and_read_only(self):
        gh = OfflineGH()
        with patch.object(op, 'PINS', gh.pins):
            result = op.preflight(gh, SOURCE)
        self.assertEqual(set(gh.controls), set(REVIEWED_PINS))
        self.assertEqual(len(gh.controls), 6)
        self.assertEqual(gh.branch_reads, 2)
        self.assertEqual(result, {'state': 'PREFLIGHT_ONLY', 'source': SOURCE,
                                  'production_authorization': False})
        self.assertTrue(all(payload is None for _, payload in gh.calls))

    def test_each_control_rejects_wrong_identity_and_tampered_bytes(self):
        for path in REVIEWED_PINS:
            for field, message in [('sha', 'native control identity changed'),
                                   ('content', 'native control bytes differ')]:
                with self.subTest(path=path, field=field):
                    gh = OfflineGH()
                    gh.bad_path, gh.bad_field = path, field
                    with patch.object(op, 'PINS', gh.pins):
                        with self.assertRaisesRegex(op.OperatorError, message):
                            op.preflight(gh, SOURCE)
                    self.assertTrue(all(payload is None for _, payload in gh.calls))
                    self.assertFalse(any('/actions/' in endpoint for endpoint, _ in gh.calls))

    def test_each_canonical_writer_still_refuses_preflight(self):
        writers = {'.github/workflows/estate-release-train.yml',
                   '.github/workflows/hf-sync.yml',
                   '.github/workflows/repair-cloudflare-product-edge.yml'}
        self.assertEqual(op.WRITERS, writers)
        for path in writers:
            with self.subTest(path=path):
                gh = OfflineGH()
                gh.active_writer = path + '@refs/heads/main'
                with patch.object(op, 'PINS', gh.pins):
                    with self.assertRaisesRegex(op.OperatorError, 'writer/controller is active'):
                        op.preflight(gh, SOURCE)
                self.assertTrue(all(payload is None for _, payload in gh.calls))

    def test_rate_limit_and_source_advance_remain_refusals(self):
        for attribute, message in [('rate_limited', 'synthetic rate limit'),
                                   ('move_on_recheck', 'approved source is no longer current main')]:
            with self.subTest(attribute=attribute):
                gh = OfflineGH()
                setattr(gh, attribute, True)
                with patch.object(op, 'PINS', gh.pins):
                    with self.assertRaisesRegex(op.OperatorError, message):
                        op.preflight(gh, SOURCE)
                self.assertTrue(all(payload is None for _, payload in gh.calls))


if __name__ == '__main__':
    unittest.main()
