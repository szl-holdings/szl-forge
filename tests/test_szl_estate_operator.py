"""Synthetic, offline operator tests; never evidence of a live deployment."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from tools import szl_estate_operator as op

SHA = 'a' * 40


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data, usedforsecurity=False).hexdigest()


class FakeGH:
    def __init__(self):
        self.calls = []
        self.moved = False
        self.active = False
        self.fail_post = False
        self.response = {'workflow_run_id': 77, 'run_url': f'https://api.github.com/repos/{op.REPO}/actions/runs/77', 'html_url': f'https://github.com/{op.REPO}/actions/runs/77'}

    def request(self, endpoint, payload=None, binary=False):
        self.calls.append((endpoint, payload))
        if payload is not None:
            if self.fail_post:
                raise op.OperatorError('transport unavailable')
            return self.response
        if endpoint.endswith('/branches/main'):
            return {'name': 'main', 'protected': True, 'commit': {'sha': 'b'*40 if self.moved else SHA}}
        if '/actions/runs?' in endpoint:
            runs = [{'path': '.github/workflows/hf-sync.yml', 'status': 'in_progress'}] if self.active else []
            return {'total_count': len(runs), 'workflow_runs': runs}
        if endpoint.endswith('/estate-release-train.yml'):
            return {'path': op.WORKFLOW_PATH, 'state': 'active', 'id': 9}
        raise AssertionError(endpoint)


def native_fixture(aligned=True):
    estate = {
        'schema': 'szl.estate-release-train.receipt/v1',
        'state': 'ALIGNED' if aligned else 'DRIFT',
        'source_vector': {key: SHA for key in op.VECTOR_REPOS},
        'blockers': [] if aligned else ['terra:SOURCE_REVISION_MISMATCH'],
        'observed_at': '2026-09-13T12:01:00Z',
        'authority': {'production_authorization': False},
        'provider_writes_performed': False,
        'components': [],
    }
    for key in op.REQUIRED:
        estate['source_vector'][key] = SHA
        estate['components'].append({'key': key, 'required': True, 'aligned': True, 'blockers': [], 'source': {'observed': True, 'sha': SHA}, 'runtime': {'observed': True, 'revision': SHA}})
    for key in ('product', 'proof', 'profile_inventory'):
        estate[key] = {'aligned': True, 'blockers': []}
    estate['product'].update(semantic_parity=True, domain_source={'observed': True, 'revision': SHA})
    if not aligned:
        estate['components'][0]['aligned'] = False
    raw = json.dumps(estate).encode()
    inv = {'schema': 'szl.public-inventory-preflight/v1', 'state': 'ALIGNED', 'estate_receipt_sha256': hashlib.sha256(raw).hexdigest(),
           'checkout_source_revision': SHA, 'estate_source_revision': SHA, 'scope_sha256': op.SCOPE,
           'scope': {'id':'hf-public-author-membership/v1','authentication':'none','visibility':'public-only','kinds':['models','datasets','spaces'],'include_gated_metadata':True,'include_disabled_metadata':True,'include_reserved_readme_if_public':True,'kernel_policy':'count-once-as-model-repository-not-a-fourth-kind','collections_and_buckets':'outside-repository-membership-scope','portfolio_and_operational_policy':False}, 'production_authorization': False,
           'comparison': {'aligned': True, 'blockers': [], 'delta': {k: {'added': [], 'removed': [], 'state': 'MATCH'} for k in ('models', 'datasets', 'spaces')}}}
    return raw, inv


def archive(raw, inv, extra=None):
    inv = copy.deepcopy(inv)
    inv['record_sha256'] = hashlib.sha256(json.dumps({k:v for k,v in inv.items() if k != 'record_sha256'},sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('root/reports/estate-release-train.json', raw)
        z.writestr('root/reports/hf-public-inventory-preflight.json', json.dumps(inv))
        if extra:
            # ZIP's Windows writer normalizes backslashes. Construct the same
            # malicious raw name on every OS instead of testing a sanitized ZIP.
            name, content = extra
            z.writestr(name.replace('\\', '/'), content)
    result = stream.getvalue()
    if extra and '\\' in extra[0]:
        wanted = extra[0].encode('ascii')
        canonical = extra[0].replace('\\', '/').encode('ascii')
        assert len(wanted) == len(canonical) and result.count(canonical) == 2
        result = result.replace(canonical, wanted)  # Local and central headers.
    return result


class OperatorTests(unittest.TestCase):
    def test_strict_json(self):
        for value in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]'):
            with self.subTest(value=value), self.assertRaises(op.OperatorError):
                op.object_json(value)

    def test_exact_source_and_protection(self):
        gh = FakeGH()
        op.check_source(gh, SHA)
        gh.moved = True
        with self.assertRaises(op.OperatorError):
            op.check_source(gh, SHA)

    def test_active_writer_refused(self):
        gh = FakeGH(); gh.active = True
        with self.assertRaises(op.OperatorError):
            op.check_idle(gh)
        self.assertFalse(any(body is not None for _, body in gh.calls))

    def test_idle_checks_all_active_states(self):
        gh = FakeGH(); op.check_idle(gh)
        self.assertEqual(len(gh.calls), len(op.ACTIVE_STATES))

    def test_dispatch_is_exact_and_journal_precedes_post(self):
        gh = FakeGH()
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / 'attempt.jsonl'
            original = gh.request
            def checked(endpoint, payload=None, binary=False):
                if payload is not None:
                    self.assertIn('DISPATCH_ATTEMPTED', journal.read_text())
                return original(endpoint, payload, binary)
            gh.request = checked
            result = op.dispatch(gh, SHA, journal)
            self.assertEqual(result['run_id'], 77)
            posts = [body for _, body in gh.calls if body is not None]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0]['ref'], 'main')
            self.assertEqual(posts[0]['inputs']['repair'], True)
            self.assertEqual(json.loads(posts[0]['inputs']['vertical_plan_json']), {'complete': True, 'approved': True, 'expected_source_revision': SHA})
            with self.assertRaises(FileExistsError):
                op.dispatch(gh, SHA, journal)
            self.assertEqual(len([body for _, body in gh.calls if body is not None]), 1)

    def test_uncertain_dispatch_is_not_retried(self):
        gh = FakeGH(); gh.fail_post = True
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / 'attempt.jsonl'
            result = op.dispatch(gh, SHA, journal)
            self.assertEqual(result['state'], 'UNKNOWN_AFTER_ATTEMPT')
            self.assertIn('UNKNOWN_AFTER_ATTEMPT', journal.read_text())
            self.assertEqual(len([x for x in gh.calls if x[1] is not None]), 1)

    def test_legacy_empty_and_wrong_response_are_unknown(self):
        for response in (None, {}, {'workflow_run_id': True}, {'workflow_run_id': 77, 'html_url': 'https://evil.invalid'}):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmp:
                gh = FakeGH(); gh.response = response
                self.assertEqual(op.dispatch(gh, SHA, Path(tmp)/'a')['state'], 'UNKNOWN_AFTER_ATTEMPT')

    def test_source_moves_before_post(self):
        gh = FakeGH(); gh.moved = True
        with tempfile.TemporaryDirectory() as tmp:
            result = op.dispatch(gh, SHA, Path(tmp)/'a')
            self.assertEqual(result['state'], 'STOPPED_BEFORE_DISPATCH')
            self.assertFalse(any(body is not None for _, body in gh.calls))

    def test_complete_native_receipt_is_not_production_authority(self):
        raw, inv = native_fixture()
        result = op.inspect_archive(archive(raw, inv), SHA)
        self.assertEqual(result['state'], 'NATIVE_ALIGNMENT_RECEIPT_VERIFIED')
        self.assertIs(result['production_authorization'], False)

    def test_real_native_drift_is_preserved(self):
        raw, inv = native_fixture(False)
        result = op.inspect_archive(archive(raw, inv), SHA)
        self.assertEqual(result['state'], 'NATIVE_RECEIPT_VERIFIED_HOLD')
        self.assertIn('terra:SOURCE_REVISION_MISMATCH', result['blockers'])

    def test_source_scope_link_and_authority_negatives(self):
        for key, value in [('checkout_source_revision', 'b'*40), ('scope_sha256','b'*64), ('estate_receipt_sha256','b'*64), ('production_authorization', True)]:
            raw, inv = native_fixture(); inv[key] = value
            with self.subTest(key=key), self.assertRaises(op.OperatorError):
                op.inspect_archive(archive(raw, inv), SHA)

    def test_missing_or_duplicate_required_component_cannot_pass(self):
        for mode in ('missing', 'duplicate', 'wrong-runtime', 'false-required'):
            raw, inv = native_fixture(); estate=json.loads(raw)
            if mode == 'missing': estate['components'].pop()
            if mode == 'duplicate': estate['components'].append(copy.deepcopy(estate['components'][0]))
            if mode == 'wrong-runtime': estate['components'][0]['runtime']['revision']='b'*40
            if mode == 'false-required': estate['components'][0]['required']=False
            raw=json.dumps(estate).encode(); inv['estate_receipt_sha256']=hashlib.sha256(raw).hexdigest()
            with self.subTest(mode=mode), self.assertRaises(op.OperatorError):
                op.inspect_archive(archive(raw, inv), SHA)

    def test_path_and_duplicate_receipt_negatives(self):
        raw, inv = native_fixture()
        for name in ('../escape.json', '/absolute.json', 'a\\b.json', 'another/estate-release-train.json'):
            with self.subTest(name=name), self.assertRaises(op.OperatorError):
                op.inspect_archive(archive(raw, inv, (name, raw)), SHA)

    def test_zip_bounds_precede_member_reads(self):
        raw, inv = native_fixture()
        with patch.object(op, 'MAX_EXPANDED', 1), patch.object(zipfile.ZipFile, 'read', side_effect=AssertionError('read')):
            with self.assertRaises(op.OperatorError): op.inspect_archive(archive(raw, inv), SHA)

    def test_only_fixed_github_repo_endpoint(self):
        gh = op.GHClient(executable='/not/executed')
        with self.assertRaises(op.OperatorError): gh.request('https://evil.invalid')

    def test_bool_id_and_unbound_run_rejected(self):
        run={'id':77,'head_sha':SHA,'head_branch':'main','event':'workflow_dispatch','path':op.WORKFLOW_PATH,'repository':{'id':op.REPO_ID},'run_attempt':1}
        op.validate_run(run, 77, SHA)
        for key, value in [('id',True),('head_sha','b'*40),('event','push'),('path','.github/workflows/other.yml'),('run_attempt',True)]:
            bad=copy.deepcopy(run); bad[key]=value
            with self.subTest(key=key), self.assertRaises(op.OperatorError): op.validate_run(bad,77,SHA)


    def test_preflight_verifies_bytes_and_never_posts(self):
        import base64
        body = b"reviewed synthetic control\n"
        pin = blob(body)
        gh = FakeGH(); original = gh.request
        def reply(endpoint, payload=None, binary=False):
            if '/contents/' in endpoint:
                path = endpoint.split('/contents/')[1].split('?')[0]
                return {'type':'file','path':path,'sha':pin,'encoding':'base64','content':base64.b64encode(body).decode()}
            return original(endpoint, payload, binary)
        gh.request = reply
        with patch.object(op, 'PINS', {op.WORKFLOW_PATH: pin}):
            self.assertEqual(op.preflight(gh,SHA)['state'],'PREFLIGHT_ONLY')
        self.assertFalse(any(payload is not None for _,payload in gh.calls))
        with patch.object(op, 'PINS', {op.WORKFLOW_PATH: 'f'*40}), self.assertRaises(op.OperatorError):
            op.preflight(gh,SHA)

    def test_incomplete_enumeration_refuses_before_any_post(self):
        gh=FakeGH()
        gh.request=lambda *a,**k: {'total_count':101,'workflow_runs':[]}
        with self.assertRaises(op.OperatorError): op.check_idle(gh)

    def test_journal_failure_prevents_mutation(self):
        gh=FakeGH()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError): op.dispatch(gh,SHA,Path(tmp)/'missing'/'file')
        self.assertFalse(any(payload is not None for _,payload in gh.calls))

    def test_scope_content_cannot_borrow_correct_digest(self):
        raw,inv=native_fixture();inv['scope']['visibility']='private'
        with self.assertRaises(op.OperatorError): op.inspect_archive(archive(raw,inv),SHA)

    def test_readonly_run_recovery_never_dispatches(self):
        gh=FakeGH()
        run={'id':77,'head_sha':SHA,'head_branch':'main','event':'workflow_dispatch','path':op.WORKFLOW_PATH,'repository':{'id':op.REPO_ID},'run_attempt':1,'status':'queued'}
        gh.request=lambda endpoint,payload=None,binary=False: (self.assertIsNone(payload) or run)
        self.assertEqual(op.observe(gh,SHA,77)['state'],'RUN_PENDING_HOLD')

    def test_native_receipt_empty_components_cannot_pass(self):
        raw,inv=native_fixture();estate=json.loads(raw);estate['components']=[]
        raw=json.dumps(estate).encode();inv['estate_receipt_sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(op.OperatorError): op.inspect_archive(archive(raw,inv),SHA)


    def readback_fixture(self):
        raw, inv = native_fixture()
        packed = archive(raw, inv)
        run = {'id':77, 'head_sha':SHA, 'head_branch':'main',
               'event':'workflow_dispatch', 'path':op.WORKFLOW_PATH,
               'repository':{'id':op.REPO_ID}, 'run_attempt':1,
               'status':'completed', 'conclusion':'success',
               'created_at':'2026-09-13T12:00:00Z', 'updated_at':'2026-09-13T12:02:00Z'}
        item = {'id':88, 'name':'estate-release-train-77-1', 'size_in_bytes':len(packed),
                'expired':False, 'digest':'sha256:'+hashlib.sha256(packed).hexdigest(),
                'workflow_run':{'id':77, 'head_sha':SHA, 'repository_id':op.REPO_ID}}
        page = {'total_count':1, 'artifacts':[item]}
        gh=FakeGH()
        def request(endpoint, payload=None, binary=False):
            gh.calls.append((endpoint,payload))
            self.assertIsNone(payload, 'readback must never mutate')
            if endpoint.endswith('/actions/runs/77'): return run
            if endpoint.endswith('/actions/runs/77/artifacts?per_page=100'): return page
            if endpoint.endswith('/actions/artifacts/88/zip'):
                self.assertTrue(binary)
                return packed
            if endpoint.endswith('/branches/main'):
                return {'name':'main', 'protected':True, 'commit':{'sha':SHA}}
            raise AssertionError(endpoint)
        gh.request=request
        return gh, run, page, item

    def test_whole_readback_binds_archive_run_clock_and_current_vector(self):
        gh, run, page, item=self.readback_fixture()
        result=op.observe(gh,SHA,77)
        self.assertEqual(result['state'],'NATIVE_ALIGNMENT_RECEIPT_VERIFIED')
        self.assertEqual(result['run_id'],77)
        self.assertFalse(result['independent_live_probe_replay'])
        self.assertFalse(result['production_authorization'])
        for repo in op.VECTOR_REPOS.values():
            self.assertIn((f'repos/szl-holdings/{repo}/branches/main',None),gh.calls)

    def test_whole_readback_rejects_outer_artifact_mismatch(self):
        for field,value in [('digest','sha256:'+'0'*64),('size_in_bytes',3),('expired',True),('id',True)]:
            gh,run,page,item=self.readback_fixture(); item[field]=value
            with self.subTest(field=field), self.assertRaises(op.OperatorError):
                op.observe(gh,SHA,77)

    def test_whole_readback_cannot_override_native_failure_or_copy_old_receipt(self):
        for field,value in [('conclusion','failure'),('created_at','2026-09-14T12:00:00Z'),('updated_at','2026-09-13T12:00:00Z'),('run_attempt',2)]:
            gh,run,page,item=self.readback_fixture(); run[field]=value
            with self.subTest(field=field), self.assertRaises(op.OperatorError):
                op.observe(gh,SHA,77)

    def test_whole_readback_refuses_moved_proof_source(self):
        gh,run,page,item=self.readback_fixture(); original=gh.request
        def moved(endpoint,payload=None,binary=False):
            if endpoint == 'repos/szl-holdings/a11oy-net/branches/main':
                return {'commit':{'sha':'b'*40}}
            return original(endpoint,payload,binary)
        gh.request=moved
        with self.assertRaisesRegex(op.OperatorError,'authority vector moved'):
            op.observe(gh,SHA,77)

    def test_gh_mutation_is_limited_to_native_dispatch(self):
        gh=op.GHClient(executable='/not/executed')
        with self.assertRaisesRegex(op.OperatorError,'only native estate dispatch'):
            gh.request(f'{op.PREFIX}/issues',{'title':'never'})

    def test_raw_backslash_fixture_survives_writer_normalization(self):
        raw,inv=native_fixture()
        data=archive(raw,inv,('a\\b.json',raw))
        self.assertEqual(data.count(b'a\\b.json'),2)
        with self.assertRaisesRegex(op.OperatorError,'unsafe archive path'):
            op.inspect_archive(data,SHA)


if __name__ == '__main__':
    unittest.main()
