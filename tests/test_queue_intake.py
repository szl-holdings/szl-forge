# SPDX-License-Identifier: Apache-2.0
"""Synthetic regression fixtures; no model load, provider write or training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('queue_intake_test', ROOT/'training/queue_intake.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def queue():
    return {'schema':'szl.training-queue-import/v1','declared_total':1,
            'entries':[{'name':'sample','base':'example/base','dataset':'example/data'}]}


def snapshot():
    common={'status':200,'sha':'a'*40,'files':['model.safetensors']}
    return [{'kind':'models','id':'SZLHOLDINGS/sample','pipeline_tag':'text-generation',
             'cardData':{'base_model':'example/base'},**common},
            {'kind':'models','id':'example/base',**common},
            {'kind':'datasets','id':'example/data',**common}]


def fixtures():
    return {'schema':'szl.exact-response-fixtures/v2','cases':[
        {'id':'a1','kind':'answerable','question':'What is 17 * 23?','accepted':['391']},
        {'id':'u1','kind':'unanswerable_under_fixture','question':'Unknown under fixture','accepted':['ABSTAIN']}]}


def response(a='391', u='ABSTAIN'):
    return {'schema':'szl.raw-smoke-responses/v2','fixture_sha256':'b'*64,
            'evidence_class':'SIMULATED','model_binding':{'status':'UNVERIFIED'},
            'responses':[{'id':ident,'status':'OK','response':text,'finish_reason':'stop'}
                         for ident,text in [('a1',a),('u1',u)]]}


class QueueTests(unittest.TestCase):
    def run_intake(self,q=None,s=None):
        return module.intake(queue() if q is None else q, snapshot() if s is None else s, config_dir=ROOT)

    def test_missing_operator_inputs_remain_rejected_by_real_contract(self):
        result=self.run_intake()
        self.assertFalse(result['training_authorized'])
        row=result['entries'][0]
        self.assertEqual(row['launch_contract']['schema'],'szl.forge.launch-contract/v1')
        self.assertEqual(row['launch_contract']['verdict'],'REJECT')
        for code in ['LAUNCH_ADMISSION_MISSING','LAUNCH_CORPUS_FLOOR','LAUNCH_HOLDOUT_MISSING']:
            self.assertIn(code,row['blockers'])
        self.assertFalse(row['launch_contract']['training_started'])
        self.assertFalse(result['hub_write'])

    def test_declared_total_cannot_hide_omitted_jobs(self):
        q=queue();q['declared_total']=59
        result=self.run_intake(q)
        self.assertFalse(result['population_matches'])
        self.assertIn('QUEUE_POPULATION_MISMATCH',result['entries'][0]['blockers'])

    def test_duplicate_target_case_alias_rejected(self):
        q=queue();q['entries'].append({**q['entries'][0],'name':'SAMPLE'})
        with self.assertRaises(module.IntakeError):self.run_intake(q)

    def test_empty_and_invalid_declared_populations_rejected(self):
        for changes in [{'entries':[]},{'declared_total':True},{'declared_total':-1}]:
            with self.subTest(changes=changes),self.assertRaises(module.IntakeError):self.run_intake({**queue(),**changes})

    def test_embeddings_do_not_enter_lm_training(self):
        for task in ['sentence-similarity','feature-extraction',None]:
            s=snapshot();s[0]['pipeline_tag']=task
            self.assertIn('TARGET_NOT_QUALIFIED_FOR_CAUSAL_LM_SFT',self.run_intake(s=s)['entries'][0]['blockers'])

    def test_base_mismatch_does_not_silently_change_supplied_base(self):
        s=snapshot();s[0]['cardData']['base_model']='other/model'
        row=self.run_intake(s=s)['entries'][0]
        self.assertIn('SUPPLIED_BASE_DIFFERS_FROM_DECLARED_LINEAGE',row['blockers'])
        self.assertEqual(row['proposed_wave']['student']['id'],'example/base')

    def test_unavailable_is_not_nonexistence(self):
        for status in [401,403,404,'UNAVAILABLE']:
            s=snapshot();s[0].update(status=status,sha=None)
            self.assertIn('TARGET_OBSERVATION_UNAVAILABLE',self.run_intake(s=s)['entries'][0]['blockers'])

    def test_floating_revision_not_proposed_as_a_pin(self):
        s=snapshot();s[1]['sha']='main'
        row=self.run_intake(s=s)['entries'][0]
        self.assertIsNone(row['proposed_wave']['student']['revision'])
        self.assertIn('BASE_OBSERVATION_UNAVAILABLE',row['blockers'])

    def test_same_name_in_different_namespaces_does_not_confuse_identity(self):
        s=snapshot();s.append({'kind':'datasets','id':'example/base','status':200,'sha':'c'*40})
        self.assertEqual(self.run_intake(s=s)['entries'][0]['proposed_wave']['student']['revision'],'a'*40)

    def test_duplicate_snapshot_rejected(self):
        s=snapshot();s.append(copy.deepcopy(s[0]))
        with self.assertRaises(module.IntakeError):self.run_intake(s=s)

    def test_mixed_fixture_and_weight_target_requires_review(self):
        s=snapshot();s[0]['files'].append('fixture.npz')
        self.assertIn('MIXED_SOFTWARE_WEIGHT_DESTINATION_REQUIRES_REVIEW',self.run_intake(s=s)['entries'][0]['blockers'])

    def test_permissive_license_never_fabricates_admission(self):
        s=snapshot();s[2]['cardData']={'license':'apache-2.0'}
        row=self.run_intake(s=s)['entries'][0]
        self.assertIsNone(row['proposed_wave']['admission'])
        self.assertFalse(row['training_authorized'])

    def test_model_stack_never_imported(self):
        import builtins
        original=builtins.__import__
        def deny(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','peft','datasets','trl','transformers','huggingface_hub','unsloth'}:
                raise AssertionError('Model stack imported')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',deny):self.run_intake()


class CharacterizationTests(unittest.TestCase):
    def score(self,r=None,f=None):
        return module.characterize(fixtures() if f is None else f, response() if r is None else r, fixture_sha256='b'*64)

    def test_correct_exact_answers_but_no_promotion(self):
        result=self.score()
        self.assertEqual(result['outcomes'],{'ACCEPTED_EXACT_RESPONSE':2})
        self.assertFalse(result['promotion_authorized'])
        self.assertFalse(result['signature_verified'])
        self.assertEqual(result['evidence_class'],'SIMULATED')

    def test_reproduces_old_wrong_answer_pass_and_rejects_it(self):
        wrong='392'
        legacy_abstained=any(x in wrong.lower() for x in ["i don't know",'ABSTAIN'])
        self.assertTrue(not legacy_abstained)
        self.assertEqual(self.score(response(a=wrong))['rows'][0]['verdict'],'INCORRECT_EXACT_ANSWER')

    def test_reproduces_old_uppercase_abstain_bug(self):
        self.assertFalse(any(x in 'ABSTAIN'.lower() for x in ["i don't know",'ABSTAIN']))
        self.assertEqual(self.score()['rows'][1]['verdict'],'ACCEPTED_EXACT_RESPONSE')

    def test_quoted_or_negated_refusal_is_not_acceptance(self):
        for value in ['I do not ABSTAIN; the answer is 12','Do not say ABSTAIN','He said I do not know; it was 50']:
            with self.subTest(value=value):
                self.assertEqual(self.score(response(u=value))['rows'][1]['verdict'],'REVIEW_REQUIRED')

    def test_refusal_on_answerable_is_preserved(self):
        self.assertEqual(self.score(response(a='ABSTAIN'))['rows'][0]['verdict'],'ABSTAINED_ON_ANSWERABLE')

    def test_missing_and_wrong_ids_rejected_even_when_counts_equal(self):
        for ids in [[],['a1'],['a1','wrong'],['a1','a1']]:
            r=response();r['responses']=[{**r['responses'][0],'id':i} for i in ids]
            with self.subTest(ids=ids),self.assertRaises(module.IntakeError):self.score(r)

    def test_truncation_and_transport_failure_not_success(self):
        for changes in [{'finish_reason':'length'},{'status':'ERROR'},{'response':None},{'response':''}]:
            r=response();r['responses'][0].update(changes)
            self.assertEqual(self.score(r)['rows'][0]['verdict'],'UNAVAILABLE')

    def test_full_response_is_retained_not_200_character_sample(self):
        text='A response requiring review. '*30
        row=self.score(response(a=text))['rows'][0]
        self.assertEqual(row['response'],text)
        self.assertEqual(row['response_sha256'],hashlib.sha256(text.encode()).hexdigest())

    def test_fixture_binding_required(self):
        r=response();r['fixture_sha256']='c'*64
        with self.assertRaises(module.IntakeError):self.score(r)

    def test_empty_duplicate_and_bad_acceptance_fixtures_rejected(self):
        for cases in [[],[fixtures()['cases'][0]]*2,[{**fixtures()['cases'][0],'accepted':[]}]]:
            f={**fixtures(),'cases':cases}
            with self.assertRaises(module.IntakeError):self.score(f=f)

    def test_whitespace_and_case_normalize_both_sides(self):
        self.assertEqual(self.score(response(' 391. ',' abstain! '))['outcomes'],{'ACCEPTED_EXACT_RESPONSE':2})

    def test_oversized_response_is_not_silently_truncated(self):
        with self.assertRaises(module.IntakeError):self.score(response(a='x'*131073))

    def test_nonliteral_explanation_is_not_assumed_semantically_correct(self):
        self.assertEqual(self.score(response(a='I think it is 391 or 400.'))['rows'][0]['verdict'],'REVIEW_REQUIRED')


class ServingMetadataTests(unittest.TestCase):
    def catalog(self):
        return {"object":"list","data":[{"id":"sample","created":100,"root":"example/base",
                "owned_by":"vllm","max_model_len":4096,"permission":[{"id":"modelperm-a",
                "created":100,"allow_sampling":True,"allow_fine_tuning":False}]}]}

    def test_ephemeral_timestamp_and_permission_id_are_not_weights(self):
        a=self.catalog();b=copy.deepcopy(a);b['data'][0]['created']=200
        b['data'][0]['permission'][0].update(id='modelperm-b',created=200)
        self.assertEqual(module.serving_metadata_projection(a),module.serving_metadata_projection(b))

    def test_actual_configuration_change_remains_visible(self):
        for key,value in [('root','different/base'),('max_model_len',8192),('id','different')]:
            a=self.catalog();b=copy.deepcopy(a);b['data'][0][key]=value
            self.assertNotEqual(module.serving_metadata_projection(a),module.serving_metadata_projection(b))

    def test_permissions_are_not_erased(self):
        a=self.catalog();b=copy.deepcopy(a);b['data'][0]['permission'][0]['allow_fine_tuning']=True
        self.assertNotEqual(module.serving_metadata_projection(a),module.serving_metadata_projection(b))

    def test_unknown_additional_field_remains_in_comparison(self):
        a=self.catalog();b=copy.deepcopy(a);b['data'][0]['adapter_digest']='different'
        self.assertNotEqual(module.serving_metadata_projection(a),module.serving_metadata_projection(b))

    def test_empty_and_duplicate_models_fail(self):
        for rows in [[],self.catalog()['data']*2]:
            with self.assertRaises(module.IntakeError):module.serving_metadata_projection({'object':'list','data':rows})

    def test_boolean_timestamp_rejected(self):
        a=self.catalog();a['data'][0]['created']=True
        with self.assertRaises(module.IntakeError):module.serving_metadata_projection(a)


class InputOutputTests(unittest.TestCase):
    def test_strict_json_negative_inputs(self):
        for raw in [b'',b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'\xff']:
            with self.subTest(raw=raw),self.assertRaises(module.IntakeError):module.decode(raw)

    def test_raw_and_escaped_credentials_are_rejected_without_echo(self):
        value='hf_'+'x'*30
        for raw in [json.dumps({'token':value}).encode(),json.dumps({'token':value}).replace('h','\\u0068').encode()]:
            with self.assertRaises(module.IntakeError) as caught:module.decode(raw)
            self.assertNotIn(value,str(caught.exception))

    def test_exclusive_receipt_write_and_digest_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'r.json';report={'status':'SIMULATED','promotion_authorized':False}
            module.write_new(p,report);original=p.read_bytes();read=json.loads(original)
            self.assertEqual(read['receipt_sha256'],module.digest(report))
            with self.assertRaises(FileExistsError):module.write_new(p,{'status':'changed'})
            self.assertEqual(p.read_bytes(),original)

    def test_cli_accepts_no_execution_mode(self):
        with self.assertRaises(SystemExit),patch.object(sys,'stderr'):
            module.main(['train'])


if __name__ == '__main__':
    unittest.main()
