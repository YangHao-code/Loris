"""Compact proof roots remain fully reconstructible and independently bound."""
import gzip,json
from pathlib import Path
import numpy as np
import pytest
from loris.document import Document
from loris._engine.rdl_reference import Doc,Label,Text,ML,Rule,RefGamma
from loris._engine.typed_runtime import IndexedChaseRunner,_plain
from loris._storage.proof_storage import make_exact_proof_runner_class
from loris._storage.state_execution import make_exact_state_runner_class
from loris._storage.discovery_factory import make_discovery_runner
from loris._storage.implicit_seed_proof import ImplicitSeedProof,load,validate_proof_roots,provenance_root_records
from loris.discovery.runner import _write_provenance
Old=make_exact_state_runner_class(make_exact_proof_runner_class())


def test_seed_records_order_lookup_count_and_immutable_ownership(tmp_path):
    rng=np.random.RandomState(2);p=rng.rand(11,4)<.3;n=(rng.rand(11,4)<.5)&~p
    gamma=RefGamma(p,n);ctx=dict(docs=[Document('x')]*11,labels=list('ABCD'),gamma=gamma)
    runner=IndexedChaseRunner(ctx,tmp_path)
    try:old=runner._seed_proof(runner.gamma)
    finally:runner.close()
    compact=ImplicitSeedProof(p,n)
    assert list(old)==list(compact) and len(old)==len(compact)
    assert old.to_dict()==compact.to_dict()
    p[:]=False;n[:]=False
    assert old.to_dict()==compact.to_dict()
    for key in ('pos:00:0','subset:1:2','pos:-1:0','neg:0:4','x:0:0','subset:1:1:0',None):
        assert key not in compact
        with pytest.raises(KeyError):compact[key]
    with pytest.raises(ValueError):compact.positive.setflags(write=True)
    with pytest.raises(TypeError):compact['subset:0:0']['origin']='changed'


@pytest.mark.parametrize('protected',[False,True])
def test_full_proof_logical_records_and_counts_match_eager_roots(tmp_path,protected):
    for seed in range(15):
        rng=np.random.RandomState(seed);n,k=10,4
        p=rng.rand(n,k)<.15;nmat=(rng.rand(n,k)<.2)&~p
        ctx=dict(docs=[Document(str(i)) for i in range(n)],labels=list('ABCD'),
            gamma=RefGamma(p,nmat,((0,2),(2,6))),model_probabilities={'M':rng.rand(n,k)})
        old=Old(ctx,tmp_path/(str(seed)+'old'),gamma_protection=protected)
        new=make_discovery_runner(ctx,tmp_path/(str(seed)+'new'),gamma_protection=protected)
        try:
            rules=[Rule((Doc('x'),ML('x','M','A',.4,dynamic=True)),Label('x','B')),
                   Rule((Doc('x'),Label('x','B')),Label('x','C'))]
            a,b=old(rules),new(rules)
            assert a.status==b.status and a.conflicts==b.conflicts and a.n_rounds==b.n_rounds
            for f in ('positive_facts','negative_facts','predictions'):np.testing.assert_array_equal(getattr(a,f),getattr(b,f))
            assert a.proof_dag.to_dict()==b.proof_dag.to_dict()
            assert list(a.proof_dag)==list(b.proof_dag)
            assert len(a.proof_dag)==len(b.proof_dag)
            assert a.provenance==b.provenance and a.reference_audit==b.reference_audit
            assert old.stats==new.stats
        finally:old.close();new.close()


def test_bitmap_roundtrip_and_tamper_fail_closed(tmp_path):
    p=np.array([[1,0],[0,0]],bool);root=ImplicitSeedProof(p,~p)
    record=root.save(tmp_path/'ROOTS.npz');restored=load(record,tmp_path)
    assert root.identity()==restored.identity() and root.to_dict()==restored.to_dict()
    report=dict(proof_identity=dict(base_identity=root.identity()),implicit_seed_roots=[record])
    assert validate_proof_roots(report,tmp_path)==[record]
    with pytest.raises(ValueError):validate_proof_roots(dict(proof_identity=report['proof_identity']),tmp_path)
    with (tmp_path/'ROOTS.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='SHA'):validate_proof_roots(report,tmp_path)


def test_provenance_retains_derived_proofs_and_complete_compressed_root_layer(tmp_path):
    p=np.zeros((7,2),bool);p[0,0]=True;n=p.copy();n[:]=False
    ctx=dict(docs=[Document('x')]*7,labels=['A','B'],gamma=RefGamma(p,n))
    runner=make_discovery_runner(ctx,tmp_path/'cache')
    try:
        result=runner([Rule((Doc('x'),Label('x','A')),Label('x','B'))])
        _write_provenance(result,tmp_path)
        with gzip.open(tmp_path/'PROVENANCE.jsonl.gz','rt') as f:records=[json.loads(x) for x in f]
        assert records[0]['schema']=='loris_complete_final_provenance_v3'
        roots=provenance_root_records(tmp_path/'PROVENANCE.jsonl.gz')
        assert len(roots)==1
        restored={r['key']:r['value'] for r in records if r['type']=='proof'}
        restored.update(load(roots[0],tmp_path).to_dict())
        assert restored==result.proof_dag.to_dict()
        assert sum(r['type']=='proof' for r in records)+roots[0]['expanded_root_count']==len(result.proof_dag)
        assert len(records)<len(result.proof_dag)+5
    finally:runner.close()


def test_completed_discovery_resume_archive_and_registry_reject_changed_root_sidecar(tmp_path):
    from tests.test_discovery_runner import inputs
    from loris.discovery.runner import run_discovery
    from loris.discovery.archive import status
    from loris.experiments.registry import check_complete_program
    out = tmp_path / 'discovery'
    run_discovery(output=out, **inputs())
    report = json.loads((out / 'PROOF_REPLAY.json').read_text())
    sidecar = Path(report['implicit_seed_roots'][0]['bitmap']['path'])
    original = sidecar.read_bytes()
    sidecar.write_bytes(original + b'changed')
    try:
        with pytest.raises(ValueError, match='SHA'):
            status(out)
        with pytest.raises(ValueError, match='SHA'):
            check_complete_program(out / 'learning/SELECTED.json', out / 'COMPLETE.json')
        with pytest.raises(ValueError, match='SHA'):
            run_discovery(output=out, resume=True, **inputs())
    finally:
        sidecar.write_bytes(original)
    assert status(out)['status'] == 'complete_verified_identity'
