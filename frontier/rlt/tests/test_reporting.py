import pytest
from frontier.rlt.runtime import canonical,digest
from frontier.rlt.reporting import render


def report():
    p={'schema':'szl.rlt.synthetic-training-evaluation.v1','production_qualified':False,'publication_eligible':False,
       'runs':[{'variant':'rlt','seed':17,'parameters':7105,'test':{'16':{'programs':128,'final_correct':100,'final_accuracy':100/128}}}]}
    return p


def encoded(p):return canonical({**p,'receipt_sha256':digest(canonical(p))})


def test_render_includes_actual_counts_and_no_script():
    page=render(encoded(report()))
    assert '100/128' in page and 'Not measured' in page and '<script' not in page


def test_bad_digest_and_inconsistent_metrics_refused():
    p=report()
    with pytest.raises(ValueError):render(canonical({**p,'receipt_sha256':'0'*64}))
    p['runs'][0]['test']['16']['final_accuracy']=1.0
    with pytest.raises(ValueError):render(encoded(p))


def test_promoted_claim_refused():
    p=report();p['production_qualified']=True
    with pytest.raises(ValueError):render(encoded(p))
