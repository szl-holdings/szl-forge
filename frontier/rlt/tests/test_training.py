"""Synthetic grammar/data and real optimizer/save/load regressions."""
from dataclasses import replace
import json
import pytest
import torch
from safetensors.torch import save
from frontier.rlt.training import Plan, oracle, programs, split_of, dataset_bytes, make_model, one_run, load_candidate
from frontier.rlt.runtime import ContinuityError, digest


def test_unknown_reset_flip_and_revocation_oracle():
    assert oracle([3,2,0,2,1,4,2,3,1]) == (2,2,0,1,1,2,2,2,1)


@pytest.mark.parametrize('op',[True,False,1.0,'1',-1,5])
def test_oracle_refuses_non_operations(op):
    with pytest.raises(ValueError): oracle([op])


def test_corpus_reproducible_and_prefix_families_do_not_cross_splits():
    datasets = {s: programs(s,16,64,211) for s in ('train','validation','test')}
    for split,rows in datasets.items():
        assert rows == programs(split,16,64,211)
        assert len(rows)==len(set(rows))
        assert all(split_of(p)==split for p in rows)
        assert all(split_of(p+tuple([0]*16))==split for p in rows)
        for other, other_rows in datasets.items():
            if other != split:
                assert not {p[:6] for p in rows} & {p[:6] for p in other_rows}
        for line in dataset_bytes(rows).splitlines():
            row=json.loads(line)
            assert tuple(row['states']) == oracle(row['operations'])


@pytest.mark.parametrize('change',[{'steps':True},{'steps':0},{'steps':2001},{'learning_rate':float('nan')},
    {'learning_rate':float('inf')},{'batch_size':65},{'seeds':(17,17)},{'seeds':(True,)},{'train_length':7}])
def test_plan_bounds(change):
    with pytest.raises(ValueError): replace(Plan(),**change)


def test_no_output_feedback_ablation_is_not_no_recurrence():
    m=make_model('no_output_feedback',17)
    assert float(m.feedback_scale)==0 and not m.feedback_scale.requires_grad
    assert m.shared_attention.k.weight.requires_grad


def test_real_optimizer_and_weights_are_reproducible_and_replayable():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p=Plan(steps=2,train_count=32,eval_count=16,seeds=(17,))
    tr=programs('train',16,32,211);v=programs('validation',16,16,211);t={16:programs('test',16,16,211)}
    a,ab=one_run('rlt',17,p,tr,v,t);b,bb=one_run('rlt',17,p,tr,v,t)
    assert a['initial_weight_sha256']!=a['weight_sha256']
    assert a['weight_sha256']==b['weight_sha256'] and ab==bb
    assert a['weight_roundtrip_max_abs_error']==a['state_roundtrip_max_abs_error']==0
    assert a['test']['16']['programs']==16
    with pytest.raises(ContinuityError): load_candidate(ab,expected_sha256='0'*64)


def test_candidate_rejects_wrong_shape_even_with_correct_digest():
    blob=save({'junk':torch.ones(1)})
    with pytest.raises(ContinuityError):load_candidate(blob,expected_sha256=digest(blob))
