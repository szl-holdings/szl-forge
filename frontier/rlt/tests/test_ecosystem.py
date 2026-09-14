"""Offline callback contract tests; actual installed estate execution runs in CI."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from frontier.rlt.ecosystem import BoundaryError, EvidenceBoundController, Principal, evidence_binding, nemo_digest
from frontier.rlt.model import TinyRLT
from frontier.rlt.runtime import digest


def hit():
    items=[{'node_id':'public:one','source':'public-source','sha256':digest(b'public-source-text')}]
    return {'ready':True,'content_access':'HANDLES_ONLY','evidence':items,
            'handles':[{'nodeId':'public:one'}],'evidence_set_sha256':nemo_digest(items),'generation_sha256':digest(b'corpus-a')}


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    current=hit();calls=[]
    principal=Principal(digest(b'principal'),digest(b'tenant'),digest(b'policy'))
    def witness(envelope):
        calls.append(deepcopy(envelope))
        return SimpleNamespace(input_hash='sha256:'+nemo_digest(envelope),decision='ALLOW',rule_version='UNIT_CALLBACK_NOT_NEMO')
    c=EvidenceBoundController(TinyRLT(),principal,retrieve=lambda q,k:deepcopy(current),witness=witness,
                              authorize=lambda p,q:p==principal and q=='allowed',formulas={})
    return c,current,calls


def test_real_runtime_reuses_prefix_and_replays_edits_and_changed_evidence():
    c,h,calls=setup()
    first=c.evaluate([0,2,1],evidence_query='allowed')
    second=c.evaluate([0,2,1,3],evidence_query='allowed')
    edited=c.evaluate([1,2,1,3],evidence_query='allowed')
    assert first['continuity_mode']=='FRESH'
    assert second['continuity_mode']=='COMPATIBLE_PREFIX_REUSED'
    assert edited['continuity_mode']=='EVIDENCE_OR_HISTORY_REPLAY'
    h['generation_sha256']=digest(b'corpus-b')
    changed=c.evaluate([1,2,1,3,3],evidence_query='allowed')
    assert changed['continuity_mode']=='EVIDENCE_OR_HISTORY_REPLAY'
    assert changed['evidence_sha256']!=edited['evidence_sha256']
    assert [x['stage'] for x in calls]==['PRE_GENERATION','POST_GENERATION']*4
    assert changed['execution_authority']=='NONE' and not changed['executed']
    assert not {'tokens','prompt','hidden_state','raw_content'} & changed.keys()


def test_authorization_precedes_retrieval_and_inference():
    c,_,calls=setup()
    c._retrieve=lambda *a,**k:pytest.fail('must not retrieve after authorization denial')
    with pytest.raises(BoundaryError):c.evaluate([0],evidence_query='denied')
    assert c._cache is None and calls==[]


@pytest.mark.parametrize('stage',['PRE_GENERATION','POST_GENERATION'])
def test_witness_refusal_never_commits_a_candidate_cache(stage):
    c,_,_=setup()
    def reject(envelope):
        return SimpleNamespace(input_hash='sha256:'+nemo_digest(envelope),decision='BLOCK' if envelope['stage']==stage else 'ALLOW',rule_version='UNIT')
    c._witness=reject
    with pytest.raises(BoundaryError):c.evaluate([0,1],evidence_query='allowed')
    assert c._cache is None


def test_witness_must_bind_exact_envelope():
    c,_,_=setup()
    c._witness=lambda e:SimpleNamespace(input_hash='sha256:'+'0'*64,decision='ALLOW',rule_version='UNIT')
    with pytest.raises(BoundaryError):c.evaluate([0],evidence_query='allowed')


@pytest.mark.parametrize('mutation',['duplicate','digest','not_ready','content','empty','bad_id'])
def test_evidence_contract_fails_closed(mutation):
    h=hit()
    if mutation=='duplicate':h['evidence']*=2;h['handles']*=2;h['evidence_set_sha256']=nemo_digest(h['evidence'])
    elif mutation=='digest':h['evidence_set_sha256']='0'*64
    elif mutation=='not_ready':h['ready']=1
    elif mutation=='content':h['content_access']='RAW'
    elif mutation=='empty':h['evidence']=[];h['handles']=[]
    elif mutation=='bad_id':h['handles'][0]['nodeId']='other'
    with pytest.raises(ValueError):evidence_binding(h)


def test_unknown_is_not_verified_even_if_model_agrees():
    c,_,_=setup()
    view=c.evaluate([4],evidence_query='allowed')
    assert view['status']=='ABSTAIN_UNKNOWN' and view['task_verified'] is False


def test_single_flight_and_packaged_corpus_only(monkeypatch):
    c,_,_=setup()
    c._lock.acquire()
    try:
        with pytest.raises(BoundaryError):c.evaluate([0],evidence_query='allowed')
    finally:c._lock.release()
    monkeypatch.setenv('SECOND_BRAIN_CORPUS','unadmitted')
    with pytest.raises(BoundaryError):EvidenceBoundController.from_installed(TinyRLT(),c._principal,authorize=lambda *a:True)
