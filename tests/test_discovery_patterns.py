"""Rich primitive migration: original helper semantics and real fit extraction."""
import ast
import inspect
import json
import os
from pathlib import Path
import sqlite3
import pickle
import gzip

import numpy as np
import pytest

from loris.discovery import patterns
from loris.document import Document
from loris.chase import GammaSeed
from loris.discovery.manifest import file_record
from loris._engine.primitive_masks import TextPrimitiveCache
from loris.predicates import predicate_from_dict


@pytest.mark.golden
def test_extraction_and_ranking_helpers_match_authenticated_sources():
    root=Path(os.environ.get('LORIS_HISTORY_ROOT',Path(__file__).resolve().parents[1])).resolve()
    sources={
        'experiments/rich_rule_discovery_bgc_20260916/patterns/build_supply.py':
            ['correlation','title_domain','select_group']}
    # The retained scalar MCC reference still matches history. Ordered parallel
    # body parsing is checked by actual inventory/mask equality in
    # test_pattern_execution_acceleration, not an obsolete AST identity claim.
    missing=[path for path in sources if not (root/path).is_file()]
    if missing:
        pytest.skip('historical comparison only: missing authenticated source assets under '
                    + str(root) + ': ' + ', '.join(missing))
    for path,names in sources.items():
        tree=ast.parse((root/path).read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                implementation = ('_select_group_reference' if node.name == 'select_group'
                                  else node.name)
                actual=ast.parse(inspect.getsource(getattr(patterns,implementation))).body[0]
                actual.name = node.name
                assert ast.dump(actual)==ast.dump(node)


def example():
    y=np.array([[1,0],[0,1]]*10,bool)
    docs=[Document(cnt=('red fox red fox jumps quickly' if row[0] else
                        'blue birds fly over water'),ttl='',mtd='') for row in y]
    return dict(docs=docs,fit_indices=np.arange(20),truth=y,negative=~y,
                labels=['red','blue'],row_ids=[str(i) for i in range(20)],
                metadata=[{} for _ in docs],cluster_assignments=np.repeat([0,1],10),
                input_binding={'synthetic_fit_fixture':'20_complete_known_only'},
                min_support=2,body_quota=2,title_quota=2,label_stride=1)


def test_real_supply_no_fake_titles_and_all_masks_match_runtime(tmp_path):
    out=tmp_path/'supply';data=example()
    result=patterns.build_supply(output=out,**data)
    assert result['status']=='complete' and result['model_inference_or_training_calls']==0
    specs=json.loads((out/'PATTERNS_BY_CLUSTER.json').read_text())
    assert all(specs.values())
    cache=TextPrimitiveCache(data['docs'],tmp_path/'lexical')
    try:
        for cluster,items in specs.items():
            with np.load(out/('cluster_%02d'%int(cluster))/'FIT_MASKS.npz') as saved:
                assert saved['masks'].shape==(len(items),20)
                for spec,expected in zip(items,saved['masks']):
                    predicate=predicate_from_dict(spec['predicate'])
                    assert predicate.attr=='cnt'
                    assert np.array_equal(cache.mask(predicate),expected)
                    assert 'head' not in spec
    finally:
        cache.close()
    with sqlite3.connect(str(out/'candidates.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM candidates').fetchone()[0]>sum(map(len,specs.values()))
    with sqlite3.connect(str(out/'title_raw.sqlite')) as db:
        assert db.execute('SELECT count(*) FROM anchors').fetchone()[0]==0
    with pytest.raises(FileExistsError):patterns.build_supply(output=out,**data)


def test_incomplete_signed_fit_is_rejected_before_any_extraction(tmp_path):
    data=example();data['negative'][0,1]=False
    with pytest.raises(ValueError,match='signed label'):
        patterns.build_supply(output=tmp_path/'invalid',**data)
    assert not (tmp_path/'invalid').exists()


def test_multilabel_entropy_uses_document_fractions_and_ignores_empty_masks():
    truth=np.array([[1,1,0],[1,0,1]],bool)
    assert patterns.label_entropy(np.ones(2,bool),truth)==pytest.approx(np.log(2))
    assert patterns.label_entropy(np.array([1,0],bool),truth)==0
    assert patterns.label_entropy(np.zeros(2,bool),truth) is None


def test_entropy_screen_filters_mixed_patterns_and_records_real_number_anchors(tmp_path):
    data=example()
    data['docs']=[Document(cnt=d.cnt+(' marker' if i%5<2 else '')+
                          (' 261' if data['truth'][i,0] else ' 400'))
                  for i,d in enumerate(data['docs'])]
    out=tmp_path/'entropy'
    patterns.build_supply(output=out,max_entropy=.01,**data)
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    assert protocol['screening']=='entropy'
    with sqlite3.connect(str(out/'candidates.sqlite')) as db:
        assert db.execute("SELECT count(*) FROM candidates WHERE reason='entropy_above_maximum'").fetchone()[0]>0
        assert db.execute('SELECT count(*) FROM entropy_screening WHERE accepted=1 AND entropy>.01').fetchone()[0]==0
    with gzip.open(out/'additional_anchors.jsonl.gz','rt') as stream:
        anchors=[json.loads(line) for line in stream]
    assert any(a['kind']=='number_span' and a['surface']=='261' for a in anchors)
    assert any(a['kind']=='tfidf_term' for a in anchors)
    with sqlite3.connect(str(out/'body_raw.sqlite')) as db:
        key=next(a['key'] for a in anchors if a['kind']=='number_span' and a['surface']=='261')
        assert db.execute('SELECT valid FROM anchor_validity WHERE key=?',(key,)).fetchone()[0]==1


def test_entropy_controls_are_validated_before_extraction(tmp_path):
    for options in ({'max_entropy':-1},{'max_entropy':float('nan')},
                    {'screening':'mcc','max_entropy':.1},{'screening':'unknown'}):
        with pytest.raises(ValueError):
            patterns.build_supply(output=tmp_path/'invalid',**example(),**options)
    assert not (tmp_path/'invalid').exists()


def test_parallel_prewarm_uses_original_masks_and_detects_tampering(tmp_path):
    from loris.discovery.prewarm import prewarm
    data=example();supply=tmp_path/'supply'
    patterns.build_supply(output=supply,**data)
    n,k=24,2;fit=np.arange(20);control=np.arange(20,24)
    truth=np.array([[1,0],[0,1]]*12,bool)
    pos=np.zeros((n,k),bool);neg=pos.copy();pos[fit]=truth[fit];neg[fit]=~truth[fit]
    prob=np.where(truth,.9,.1).astype(float)
    context=dict(docs=data['docs']+data['docs'][:4],labels=data['labels'],row_ids=[str(i) for i in range(n)],
        row_roles=['source_fit']*20+['val_ctrl']*4,fit_indices=fit,receiver_indices=control,
        gamma=GammaSeed(pos,neg),truth=truth[control],source_gold={'fit':truth[fit],'val_ctrl':truth[control]},
        model_probabilities={'m':prob},model_decisions={'m':prob>=.5},context_id='prewarm_fixture24',
        attribute_values={},attribute_presence={'cnt':np.ones(n,bool),'ttl':np.zeros(n,bool),'mtd':np.zeros(n,bool)})
    with (tmp_path/'context.pkl').open('wb') as stream:pickle.dump(context,stream)
    receipt=tmp_path/'context.json'
    receipt.write_text(json.dumps({'status':'complete','context':file_record(tmp_path/'context.pkl')}))
    result=prewarm(context_receipt=receipt,supply=supply,output=tmp_path/'warm',workers=2,memory_mib=2,disk_mib=16)
    assert result['selected_fit_masks_independently_equal_to_original_full_cache']>0
    assert result['training_calls']==result['Chase_calls']==0
    masks_path=supply/'cluster_00/FIT_MASKS.npz'
    with np.load(masks_path) as source:masks=source['masks'].copy()
    masks[0,0]=~masks[0,0]
    np.savez_compressed(masks_path,masks=masks,fit_indices=fit)
    with pytest.raises(ValueError,match='mask differs'):
        prewarm(context_receipt=receipt,supply=supply,output=tmp_path/'bad',workers=1)
