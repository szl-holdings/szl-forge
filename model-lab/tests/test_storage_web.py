"""Actual Python route and packaging seams; no model download or real corpus."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from szl_model_lab.app import Settings,create_app
from szl_model_lab.cli import main
from szl_model_lab.storage_web import register_storage
from szl_model_lab import blueprints

TOKEN='test-only-local-token-not-a-real-key-32chars'
ROOT=Path(__file__).parents[1]

@pytest.fixture
def client():
    app=create_app(Settings(TOKEN,{},{}))
    with TestClient(app,base_url='http://127.0.0.1') as value:
        value.auth=('operator',TOKEN)
        yield value


@pytest.mark.parametrize('path',['/storage','/api/storage/plan'])
def test_new_views_require_parent_auth(client,path):
    assert client.get(path,auth=None).status_code==401
    assert client.get(path,auth=('other',TOKEN)).status_code==401
    assert client.get(path).status_code==200


def test_actual_route_computes_pure_arithmetic_and_keeps_unknowns(client):
    with patch('socket.create_connection',side_effect=AssertionError('no upstream')):
        response=client.get('/api/storage/plan?tokens=1048576&batch=2')
    data=response.json()
    assert data['kv_full_allocation_bytes']==288*(1<<30)
    assert data['fits_declared_budget'] is None
    assert data['observed_free_memory_bytes'] is None
    assert not any(data['authority'].values())
    assert response.headers['cache-control']=='no-store'
    assert response.headers['x-content-type-options']=='nosniff'


@pytest.mark.parametrize('query',['tokens=0','tokens=-1','tokens=1.5','tokens=1048577',
                                 'tokens=true','tokens=9&tokens=10','batch=129',
                                 'batch=01','file=/etc/passwd','tokens=10&batch=1&x=2',
                                 'tokens=%3Cscript%3E'])
def test_bounded_query_contract(client,query):
    for path in ['/storage','/api/storage/plan']:
        assert client.get(path+'?'+query).status_code==422


@pytest.mark.parametrize('method',['post','put','patch','delete'])
def test_new_views_have_no_mutating_methods(client,method):
    for path in ['/storage','/api/storage/plan']:
        assert getattr(client,method)(path).status_code==405


def test_template_renders_bound_values_and_existing_navigation(client):
    response=client.get('/storage?tokens=131072')
    assert '4.553' in response.text and '18.000' in response.text
    assert 'name="tokens"' in response.text and 'method="get"' in response.text
    assert 'value="131072"' in response.text
    assert 'src="http' not in response.text and '<script' not in response.text
    assert '/storage' in client.get('/').text
    assert client.get('/api/catalog').status_code==200
    assert client.get('/api/corpus-review').status_code==503
    assert client.get('/api/nodes').json()['nodes']==[]


def test_host_and_csp_preserved(client):
    assert client.get('/storage',headers={'host':'evil.example'}).status_code==400
    assert "default-src 'none'" in client.get('/storage').headers['content-security-policy']
    assert "frame-ancestors 'none'" in client.get('/storage').headers['content-security-policy']


def test_duplicate_registration_stops_without_partial_routes():
    app=create_app(Settings(TOKEN,{},{}));n=len(app.routes)
    with pytest.raises(ValueError):register_storage(app,lambda:None,None)
    assert len(app.routes)==n


def test_actual_cli_default_does_not_read_config_or_launch_network(capsys):
    with patch('socket.create_connection',side_effect=AssertionError('network')):
        assert main(['storage-plan','--tokens','1048576'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['kv_full_allocation_bytes']==144*(1<<30)
    assert data['config_sha256'] is None


def test_actual_cli_reads_only_supplied_digest_bound_config(tmp_path,capsys):
    from test_storage import config
    raw=json.dumps(config()).encode();path=tmp_path/'config.json';path.write_bytes(raw)
    args=['storage-plan','--tokens','32','--config',str(path),'--config-sha256',hashlib.sha256(raw).hexdigest(),'--model-revision','a'*40]
    assert main(args)==0
    assert json.loads(capsys.readouterr().out)['model_revision_declared']=='a'*40
    path.write_bytes(b'{}');assert main(args)==2
    capture=capsys.readouterr();assert capture.out=='' and str(path) not in capture.err


def test_incomplete_cli_binding_fails(capsys):
    assert main(['storage-plan','--model-revision','a'*40])==2
    assert capsys.readouterr().out==''


def test_blueprint_closure_includes_modules_ui_and_docs():
    wanted={'src/szl_model_lab/storage.py','src/szl_model_lab/storage_web.py',
            'src/szl_model_lab/templates/storage.html','docs/STORAGE_AWARE_RESEARCH.md'}
    assert wanted.issubset(blueprints.SOURCE_FILES)
    assert len(blueprints.SOURCE_FILES)==len(set(blueprints.SOURCE_FILES))
    assert all((ROOT/path).is_file() for path in wanted)

@pytest.mark.parametrize('track',['router','invariant'])
def test_actual_git_export_contains_importable_storage_stack(tmp_path,track):
    """Real Git-object export; absent ancillary docs are explicit synthetic fixtures."""
    import subprocess
    import sys
    root=tmp_path/'source';root.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
    for relative in blueprints.SOURCE_FILES:
        path=root/'model-lab'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        source=ROOT/relative
        path.write_bytes(source.read_bytes() if source.exists() else b'SYNTHETIC ANCILLARY EXPORT FIXTURE - NOT FOR PUBLICATION\n')
    git('init');git('config','user.name','Offline Test');git('config','user.email','fixture@example.invalid')
    git('add','.');git('commit','-m','Synthetic exact-object export fixture')
    revision=git('rev-parse','HEAD')
    payload=blueprints.source_payload(root,revision,track)
    binding=json.loads(payload['source-binding.json'])
    assert binding['weights_present'] is False
    assert binding['model_qualified'] is False
    for path,sha in binding['files_sha256'].items():
        assert hashlib.sha256(payload[path]).hexdigest()==sha
    assert all(not name.endswith(('.safetensors','.jsonl','.env','.joblib')) for name in payload)
    exported=tmp_path/'projection'
    for name,body in payload.items():
        path=exported/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    environment=__import__('os').environ.copy()
    environment['PYTHONPATH']=str(exported/'source/src')
    command="from szl_model_lab.app import Settings,create_app; a=create_app(Settings('x'*40,{},{})); assert any(getattr(r,'path',None)=='/storage' for r in a.routes); print('EXPORTED_STORAGE_ROUTE_IMPORT_PASSED')"
    done=subprocess.run([sys.executable,'-B','-c',command],env=environment,cwd=tmp_path,
                        capture_output=True,text=True,timeout=30,check=True)
    assert done.stdout.strip()=='EXPORTED_STORAGE_ROUTE_IMPORT_PASSED'


def test_existing_publication_contract_guards_retained():
    import ast
    # Native CI has no local reconstruction baseline. Function invariants below
    # are always checked; byte comparison is performed separately in evidence.
    source=(ROOT/'src/szl_model_lab/blueprints.py').read_text()
    functions={n.name: n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    assert {'source_payload','publish_context','publish_payload','main'}.issubset(functions)
    with pytest.raises(ValueError):blueprints.publish_context({},'a'*40)
    assert 'parent_commit=info.sha' in source
    assert 'existing_nonblueprint_artifacts_preserved' in source
    assert 'publication_byte_mismatch' in source
    assert 'protected_main_dispatch_context_required' in source
