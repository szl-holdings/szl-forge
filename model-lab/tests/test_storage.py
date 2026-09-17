"""Local synthetic CPU/filesystem controls; never upstream performance results."""
import concurrent.futures
import hashlib
import json

import pytest
import torch
from safetensors.torch import save

from szl_model_lab import storage as S


def config():
    return {"model_type": "spark2_5", "architectures": ["Spark2_5ForCausalLM"],
            "num_hidden_layers": 36, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 512,
            "max_position_embeddings": 1048576,
            "layer_types": (["sliding_attention"]*3+["full_attention"])*9,
            "auto_map": {"AutoModelForCausalLM": "untrusted.module.DoNotExecute"}}


def parse(value):
    raw = json.dumps(value).encode()
    return S.SparkShape.from_config(raw, hashlib.sha256(raw).hexdigest())


def write_store(root, transform=None):
    root.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(241)
    tensors, chunks = {}, []
    for i in range(4):
        item = {"gate": torch.randn(8, 4, generator=gen),
                "up": torch.randn(8, 4, generator=gen),
                "down": torch.randn(4, 8, generator=gen)}
        if transform:
            transform(item)
        raw = save(item)
        digest = hashlib.sha256(raw).hexdigest()
        (root/(digest+'.safetensors')).write_bytes(raw)
        key = f'expert_{i}'
        chunks.append(S.ExpertChunk(key, digest, len(raw)))
        tensors[key] = item
    return S.ExpertStore(root, S.bundle_digest(chunks), chunks), tensors


@pytest.fixture
def store(tmp_path):
    return write_store(tmp_path/'experts')


@pytest.mark.parametrize('tokens,bounded,full', [
    (32768,1.177734375,4.5), (131072,4.552734375,18.0),
    (1048576,36.052734375,144.0)])
def test_reference_kv_arithmetic(tokens,bounded,full):
    result = S.reference_plan(tokens)
    assert result['kv_if_bounded_sliding_bytes'] == int(bounded*S.GIB)
    assert result['kv_full_allocation_bytes'] == int(full*S.GIB)
    assert result['conservative_required_bytes'] is None
    assert result['fits_declared_budget'] is None
    assert result['observed_free_memory_bytes'] is None
    assert result['model_revision_declared'] is None
    assert set(result['authority'].values()) == {False}
    assert result['state'] == 'CALCULATED_NOT_QUALIFIED'


def test_config_parser_does_not_import_remote_code(monkeypatch):
    import builtins
    real = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('untrusted'): raise AssertionError('remote custom code')
        return real(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert parse(config()).kv_heads == 4


@pytest.mark.parametrize('key,value', [
    ('model_type','qwen3_5_moe'),('architectures',['ArbitraryClass']),
    ('num_hidden_layers',True),('num_hidden_layers',35),('num_attention_heads',15),
    ('num_key_value_heads',0),('head_dim',True),('head_dim',0),('head_dim',2048),
    ('sliding_window',0),('max_position_embeddings',0),('num_experts',256),
    ('layer_types',['unknown']*36),('layer_types','sliding_attention'),
    ('ignored',float('inf'))])
def test_config_bounds_and_architecture_rejected(key,value):
    d=config();d[key]=value
    with pytest.raises(ValueError): parse(d)


@pytest.mark.parametrize('raw', [b'', b'{}'+b' '*65536, b'{"a":1,"a":2}',
                                 b'{"a":NaN}', b'{"a":1e999}', b'\xff', b'['*1500])
def test_invalid_config_bytes(raw):
    with pytest.raises(ValueError): S.SparkShape.from_config(raw,hashlib.sha256(raw).hexdigest())


def test_config_hash_is_external_binding_not_assumed_provider_readback():
    raw=json.dumps(config()).encode()
    with pytest.raises(ValueError):S.SparkShape.from_config(raw,'0'*64)
    with pytest.raises(ValueError):S.SparkShape.from_config(raw,'main')


@pytest.mark.parametrize('options', [dict(tokens=True),dict(tokens=0),dict(tokens=1048577),
                                    dict(tokens=20,batch=True),dict(tokens=20,batch=129),
                                    dict(tokens=20,bytes_per_element=True),dict(tokens=20,bytes_per_element=3),
                                    dict(tokens=20,cache_layout='masked-therefore-free')])
def test_kv_bounds(options):
    with pytest.raises(ValueError):parse(config()).kv_bytes(**options)


def test_batch_scales_kv_and_small_context_uses_same_slots():
    shape=parse(config())
    assert shape.kv_bytes(128,cache_layout='bounded_sliding')==shape.kv_bytes(128)
    assert shape.kv_bytes(32768,batch=2)==2*shape.kv_bytes(32768)


def test_declared_budget_is_not_runtime_qualification():
    plan=S.memory_plan(parse(config()),tokens=32768,resident_weight_bytes=8*S.GIB,
                       runtime_reserve_bytes=2*S.GIB,memory_budget_bytes=16*S.GIB)
    assert plan['fits_declared_budget'] is True
    assert plan['conservative_required_bytes']==int(14.5*S.GIB)
    assert plan['engine_qualified'] is False
    plan=S.memory_plan(parse(config()),tokens=1048576,resident_weight_bytes=8*S.GIB,
                       runtime_reserve_bytes=2*S.GIB,memory_budget_bytes=16*S.GIB)
    assert plan['fits_declared_budget'] is False


@pytest.mark.parametrize('options',[dict(resident_weight_bytes=True),dict(memory_budget_bytes=-1),
                                    dict(runtime_reserve_bytes=1<<51),dict(model_revision='main'),
                                    dict(config_sha256='A'*64)])
def test_bad_declarations(options):
    with pytest.raises(ValueError):S.memory_plan(parse(config()),tokens=12,**options)


def test_manifest_digest_binds_content_name_size_and_not_list_order(store):
    source,_=store;chunks=list(source.chunks.values())
    assert S.bundle_digest(chunks)==S.bundle_digest(chunks[::-1])
    changed=[S.ExpertChunk('renamed',chunks[0].sha256,chunks[0].size),*chunks[1:]]
    assert S.bundle_digest(changed)!=source.bundle_sha256
    with pytest.raises(ValueError):S.ExpertStore(source.root,'0'*64,chunks)
    with pytest.raises(ValueError):S.bundle_digest(chunks+[chunks[0]])
    with pytest.raises(TypeError):source.chunks['new']=chunks[0]


@pytest.mark.parametrize('identity', ['../x','/absolute','x/y','x\\y','', 'x'*65, True])
def test_unsafe_expert_identity(identity):
    with pytest.raises(ValueError): S.ExpertChunk(identity,'a'*64,16)


def test_corrupt_file_is_not_loaded_or_cached(store):
    source,_=store;chunk=source.chunks['expert_0']
    path=source.root/(chunk.sha256+'.safetensors');b=path.read_bytes();path.write_bytes(b[:-1]+bytes([b[-1]^1]))
    cache=S.ExactExpertCache(source,chunk.size*2)
    with pytest.raises(ValueError): cache.stage(['expert_0'])
    assert cache.stats()['resident_serialized_bytes']==0
    assert cache.stats()['file_reads']==0


def test_truncated_and_symlink_files_rejected(store,tmp_path):
    source,_=store;chunk=source.chunks['expert_0'];path=source.root/(chunk.sha256+'.safetensors')
    raw=path.read_bytes();path.write_bytes(raw[:-1])
    with pytest.raises(ValueError):source.read('expert_0')
    path.unlink();other=tmp_path/'outside';other.write_bytes(raw);path.symlink_to(other)
    with pytest.raises(ValueError):source.read('expert_0')


@pytest.mark.parametrize('hints',[[],['expert_0'],['expert_3'],['expert_2','expert_3']])
def test_exact_forward_matches_resident_despite_wrong_hints(store,hints):
    source,tensors=store;chunk=source.chunks['expert_0'];cache=S.ExactExpertCache(source,chunk.size*2)
    cache.stage(hints)
    x=torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]])
    route=[('expert_0',.25),('expert_1',.75)]
    expected=torch.zeros_like(x)
    for key,weight in route:
        t=tensors[key];expected.add_((torch.nn.functional.silu(x@t['gate'].T)*(x@t['up'].T))@t['down'].T,alpha=weight)
    actual=S.streamed_moe_reference(cache,x,route,bundle_sha256=source.bundle_sha256)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    assert cache.stats()['peak_serialized_bytes']<=cache.capacity
    assert cache.stats()['gpu_peak_bytes'] is None
    assert cache.stats()['async_overlap_measured'] is False


def test_stage_uses_spare_space_without_evicting_demand(store):
    source,_=store;size=source.chunks['expert_0'].size;cache=S.ExactExpertCache(source,size)
    with cache.lease(['expert_0']):pass
    cache.stage(['expert_1'])
    with cache.lease(['expert_0']):pass
    assert cache.stats()['file_reads']==1 and cache.stats()['demand_cache_hits']==1
    assert cache.stats()['hints_skipped_for_budget']==1


def test_oversized_route_fails_without_reducing_expert_set(store):
    source,_=store;cache=S.ExactExpertCache(source,source.chunks['expert_0'].size)
    with pytest.raises(ValueError,match='route_exceeds'):
        with cache.lease(['expert_0','expert_1']):pass
    assert cache.stats()['file_reads']==0


def test_unknown_hints_validated_before_any_io(store):
    source,_=store;cache=S.ExactExpertCache(source,4096)
    with pytest.raises(ValueError):cache.stage(['expert_0','missing'])
    assert cache.stats()['file_reads']==0


@pytest.mark.parametrize('hints',[['expert_0','expert_0'],['expert_0']*65,'expert_0',[True]])
def test_bounded_hints(store,hints):
    source,_=store
    with pytest.raises(ValueError):S.ExactExpertCache(source,4096).stage(hints)


def test_pinned_nested_lease_is_not_evicted(store):
    source,_=store;size=source.chunks['expert_0'].size;cache=S.ExactExpertCache(source,size)
    with cache.lease(['expert_0']) as first:
        with pytest.raises(ValueError,match='leased_payloads'):
            with cache.lease(['expert_1']):pass
        assert first['expert_0']==source.read('expert_0')
    with cache.lease(['expert_1']):pass
    assert cache.stats()['peak_serialized_bytes']==size


def test_failure_releases_pins_without_replacing_old_store_bytes(store):
    source,_=store;size=source.chunks['expert_0'].size;cache=S.ExactExpertCache(source,size*2)
    bad=source.chunks['expert_1'];(source.root/(bad.sha256+'.safetensors')).unlink()
    with pytest.raises(OSError):
        with cache.lease(['expert_0','expert_1']):pass
    with cache.lease(['expert_2','expert_3']):pass
    assert cache.stats()['peak_serialized_bytes']<=size*2


def test_session_instances_never_reuse_each_others_cache(store):
    source,_=store;a=S.ExactExpertCache(source,4096);b=S.ExactExpertCache(source,4096)
    a.stage(['expert_0']);assert b.stats()['file_reads']==0
    with b.lease(['expert_0']):pass
    assert b.stats()['file_reads']==1


def test_concurrent_leases_are_serialized_and_bounded(store):
    source,_=store;size=source.chunks['expert_0'].size;cache=S.ExactExpertCache(source,size*2)
    def use(index):
        key='expert_'+str(index%4)
        with cache.lease([key]) as blobs:return hashlib.sha256(blobs[key]).hexdigest()==source.chunks[key].sha256
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(use,range(24)))
    assert cache.stats()['peak_serialized_bytes']<=size*2


@pytest.mark.parametrize('route', [[('expert_0',.1)],[('expert_0',float('nan'))],
                                  [('expert_0',True)],[('expert_0',-.1)],
                                  [('expert_0',.5),('expert_0',.5)]])
def test_invalid_execution_route_rejected(store,route):
    source,_=store
    with pytest.raises(ValueError):S.streamed_moe_reference(S.ExactExpertCache(source,4096),torch.ones(1,4),route,bundle_sha256=source.bundle_sha256)


def test_cross_bundle_execution_refused(store):
    source,_=store;cache=S.ExactExpertCache(source,4096)
    with pytest.raises(ValueError):S.streamed_moe_reference(cache,torch.ones(1,4),[('expert_0',1.0)],bundle_sha256='0'*64)
    assert cache.stats()['file_reads']==0


@pytest.mark.parametrize('kind',['bad_dtype','bad_shape','extra','nonfinite'])
def test_unsupported_or_nonfinite_expert_layout_refused(tmp_path,kind):
    def modify(t):
        if kind=='bad_dtype':t['gate']=t['gate'].to(torch.float16)
        elif kind=='bad_shape':t['gate']=torch.ones(8,5)
        elif kind=='extra':t['executable-looking']=torch.ones(1)
        else:t['up'][0,0]=float('inf')
    source,_=write_store(tmp_path/'experts',modify)
    with pytest.raises(ValueError):S.streamed_moe_reference(S.ExactExpertCache(source,4096),torch.ones(1,4),[('expert_0',1.0)],bundle_sha256=source.bundle_sha256)
