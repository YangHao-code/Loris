"""Exact matching integration, admission, bounded storage and persisted reuse."""
import json
import re

import numpy as np
import pytest

from loris.document import Document
from loris._engine.pattern_anchors import word_alternatives_regex
from loris._engine.pattern_discovery import CompleteRegexStore
from loris._engine.primitive_masks import TextPrimitiveCache
from loris._engine.shared_literal_stats import SharedLiteralStats, canonical_forms, configuration
from loris._learning.semantic_patterns import _spec
from loris.predicates._core import _Pattern, predicate_from_dict


def anchor(words, flags=int(re.I)):
    raw, groups = word_alternatives_regex([words])
    return dict(raw=raw, flags=flags, literal_alternatives_by_token=groups)


def reference(texts, a):
    result = np.zeros((2,len(texts)),np.int64);result[1] = -1
    for row,text in enumerate(texts):
        matches=list(re.finditer(a['raw'],text,a['flags']))
        result[:,row] = len(matches), matches[0].start() if matches else -1
    return result


@pytest.mark.parametrize('flags',[0,int(re.U),int(re.I),int(re.I|re.U)])
def test_ascii_case_alternatives_duplicates_and_boundaries(flags):
    texts=['Bank bank BANK banks','bank_bank bank1 xbank bank-x','','bank bank','bank bank']
    engine=SharedLiteralStats({'cnt':texts},config=configuration('always',1<<20))
    for forms in [['bank'],['banks','Bank','BANK','bank'],['bank1','bank_bank']]:
        a=anchor(forms,flags)
        got=engine.maybe_counts('cnt',a['raw'],a['flags'],metadata=a['literal_alternatives_by_token'])
        np.testing.assert_array_equal(got,reference(texts,a))
    assert engine.stats['shared_literal_builds']==1
    assert engine.stats['shared_literal_hits']==3


@pytest.mark.parametrize('special,ascii_letter',[('İ','i'),('ı','i'),('ſ','s'),('K','k')])
def test_unicode_four_special_letters_are_original_regex_fallback(special,ascii_letter):
    texts=[special,ascii_letter,special+ascii_letter,ascii_letter+'é','中'+ascii_letter,
           ascii_letter+'\u0301',' '+special+' ']
    engine=SharedLiteralStats({'cnt':texts},config=configuration('always',1<<20))
    for flags in [0,int(re.I)]:
        a=anchor([ascii_letter],flags)
        got=engine.maybe_counts('cnt',a['raw'],a['flags'])
        np.testing.assert_array_equal(got,reference(texts,a))
    assert engine.stats['shared_literal_nonascii_regex_rows']==12


@pytest.mark.parametrize('value',[
    dict(raw=r'bank',flags=2),dict(raw=r'\bbank\s+market\b',flags=2),
    dict(raw=r'(?=bank)',flags=0),dict(raw=r'\b(bank)\b',flags=2),
    dict(raw=r'\b(?:bänk)\b',flags=2),dict(raw=r'\b(?:a\-b)\b',flags=2),
    dict(raw=r'\b(?:bank)\b',flags=int(re.I|re.A)),
    dict(raw=r'\b(?:bank)\b',flags=int(re.I|re.M))])
def test_unsupported_syntax_never_uses_shared_path(value):
    assert canonical_forms(value['raw'],value['flags']) is None
    engine=SharedLiteralStats({'cnt':['bank market','bank']},config=configuration('always',1<<20))
    assert engine.maybe_counts('cnt',value['raw'],value['flags']) is None
    assert not engine.indices


def test_metadata_is_checked_and_plain_canonical_patternspec_can_be_recognized():
    a=anchor(['bank'])
    assert canonical_forms(a['raw'],a['flags'])==('bank',)
    assert canonical_forms(a['raw'],a['flags'],[['market']]) is None
    assert canonical_forms(a['raw'],a['flags'],None) is None
    assert canonical_forms(a['raw'],a['flags'],[['bank'],['market']]) is None


def test_store_auto_defers_rare_patterns_and_activates_reused_dense_batch(tmp_path):
    words=['word%d'%i for i in range(64)]
    rare=CompleteRegexStore([Document(cnt=w) for w in words],tmp_path/'rare',regex_index='literal',shared_literal_policy='auto')
    dense_text=' '.join(words)
    dense=CompleteRegexStore([Document(cnt=dense_text)]*64,tmp_path/'dense',regex_index='literal',shared_literal_policy='auto')
    try:
        for word in words:
            a=anchor([word])
            np.testing.assert_array_equal(rare.counts_first('cnt',a),reference(words,a))
            np.testing.assert_array_equal(dense.counts_first('cnt',a),reference([dense_text]*64,a))
        assert rare.stats['shared_literal_builds']==0
        assert rare.stats['shared_literal_auto_deferred']==64
        assert dense.stats['shared_literal_builds']==1
        assert dense.stats['shared_literal_hits']==33
        assert dense.stats['distinct_text_scans']==31
    finally:
        rare.close();dense.close()


def test_memory_abort_is_immediate_and_not_retried(tmp_path):
    texts=[' '.join('word%d'%i for i in range(300))]
    engine=SharedLiteralStats({'cnt':texts},config=configuration('always',1024))
    for i in range(10):
        a=anchor(['word%d'%i])
        assert engine.maybe_counts('cnt',a['raw'],a['flags']) is None
    assert engine.stats['shared_literal_builds_over_budget']==1
    assert engine.stats['shared_literal_token_occurrences']<300
    assert engine.bytes==0 and not engine.indices and len(engine.blocked)==1
    cache=TextPrimitiveCache([Document(cnt=texts[0])],tmp_path,shared_literal_policy='always',shared_literal_bytes=0)
    try:
        a=anchor(['word1'])
        np.testing.assert_array_equal(cache.regex_stats('cnt',_Pattern(a['raw'],a['flags'])),reference(texts,a))
        assert cache.stats['shared_literal_builds_over_budget']==1
        assert cache.stats['regex_scans']==1
    finally: cache.close()


def test_shared_budget_is_total_across_fields_and_case_modes():
    engine=SharedLiteralStats({'cnt':['bank BANK'], 'ttl':['market MARKET']},config=configuration('always',1300))
    for attr,word in [('cnt','bank'),('ttl','market')]:
        for flags in [0,int(re.I)]:
            a=anchor([word],flags)
            got=engine.maybe_counts(attr,a['raw'],a['flags'])
            if got is not None:np.testing.assert_array_equal(got,reference(engine.values[attr],a))
            assert engine.bytes<=1300
    assert engine.blocked


def test_complete_store_persistent_cache_precedes_any_shared_build(tmp_path):
    docs=[Document(cnt='bank BANK'),Document(cnt='K bank')];a=anchor(['bank'])
    first=CompleteRegexStore(docs,tmp_path,regex_index='literal',shared_literal_policy='always')
    first_id=first.identity
    expected=first.counts_first('cnt',a).copy();assert first.stats['shared_literal_builds']==1
    first.close()
    second=CompleteRegexStore(docs,tmp_path,regex_index='literal',shared_literal_policy='always')
    try:
        assert second.identity==first_id
        np.testing.assert_array_equal(second.counts_first('cnt',a),expected)
        assert second.stats['cache_hits']==1 and second.stats['shared_literal_builds']==0
    finally:second.close()
    off=CompleteRegexStore(docs,tmp_path,regex_index='literal',shared_literal_policy='off')
    try:assert off.identity!=first_id
    finally:off.close()


@pytest.mark.parametrize('family',['Match','Freq','Cooccur','Before'])
def test_real_serialized_predicates_empty_missing_and_all_rows(tmp_path,family):
    texts=['','banks bank market','market bank','BANK market','banké market','İ bank','banks bank market']
    docs=[Document(cnt=t) for t in texts]
    presence=np.array([False,True,True,True,True,True,False])
    left,right=anchor(['bank','banks']),anchor(['market'])
    spec=_spec(family,'cnt',left,right if family in ('Cooccur','Before') else None,
               op='>=' if family=='Freq' else None,eta=2 if family=='Freq' else None)
    predicate=predicate_from_dict(json.loads(json.dumps(spec.to_dict()))['predicate'])
    fast=TextPrimitiveCache(docs,tmp_path/'fast',{'cnt':presence},shared_literal_policy='always')
    slow=TextPrimitiveCache(docs,tmp_path/'slow',{'cnt':presence},shared_literal_policy='off')
    try:
        result=fast.mask(predicate)
        np.testing.assert_array_equal(result,slow.mask(predicate))
        np.testing.assert_array_equal(result,np.array([predicate(d) for d in docs])&presence)
        assert not result.flags.writeable
        assert fast.stats['shared_literal_hits']>0
        assert fast.stats['regex_scans']==0
        # Persistent vectors/masks stay reusable without rebuilding token index.
        fast.close();fast=TextPrimitiveCache(docs,tmp_path/'fast',{'cnt':presence},shared_literal_policy='always')
        np.testing.assert_array_equal(fast.mask(predicate),result)
        assert not fast.shared_literals.indices
    finally:fast.close();slow.close()


def test_all_frequency_operators_explicit_empty_field(tmp_path):
    from loris.predicates import FreqPredicate
    texts=['','','bank','bank bank'];present=np.array([False,True,True,True]);a=anchor(['bank'])
    cache=TextPrimitiveCache([Document(cnt=t) for t in texts],tmp_path,{'cnt':present},shared_literal_policy='always')
    try:
        for op,fn in [('==',np.equal),('!=',np.not_equal),('<',np.less),('<=',np.less_equal),('>',np.greater),('>=',np.greater_equal)]:
            for eta in range(4):
                pred=FreqPredicate('cnt',re.compile(a['raw'],a['flags']),op,eta)
                np.testing.assert_array_equal(cache.mask(pred),fn(np.array([0,0,1,2]),eta)&present)
    finally:cache.close()


def test_random_ascii_unicode_mixed_and_named_union_counterexample():
    rng=np.random.RandomState(41);vocabulary=['bank','Bank','BANK','a','ab','i','k','_x','word1']
    gaps=[' ','\t','-','\n','é','İ','中','\u0301']
    for _ in range(60):
        texts=[''.join(str(rng.choice(vocabulary))+str(rng.choice(gaps)) for _ in range(int(rng.randint(0,20)))) for _ in range(12)]
        texts+=['bank market',texts[0]]
        engine=SharedLiteralStats({'cnt':texts},config=configuration('always',1<<20))
        for _ in range(20):
            a=anchor(rng.choice(vocabulary,size=int(rng.randint(1,5))).tolist(),int(rng.choice([0,2,32,34])))
            got=engine.maybe_counts('cnt',a['raw'],a['flags'])
            np.testing.assert_array_equal(got,reference(texts,a))
    text='bank market grows'
    patterns=[r'\bbank\b',r'\bbank\s+market\b',r'\bmarket\s+grows\b']
    assert [next(re.finditer(p,text)).start() for p in patterns]==[0,0,5]
    assert [m.lastgroup for m in re.finditer('|'.join('(?P<p%d>%s)'%(i,p) for i,p in enumerate(patterns)),text)]==['p0','p2']


def test_explicit_environment_policy_and_invalid_values(monkeypatch):
    monkeypatch.setenv('LORIS_SHARED_LITERAL_POLICY','off')
    monkeypatch.setenv('LORIS_SHARED_LITERAL_BYTES','4096')
    assert configuration()['policy']=='off' and configuration()['memory_bytes']==4096
    assert configuration('always',0)['memory_bytes']==0
    with pytest.raises(ValueError):configuration('approximately')
    with pytest.raises(ValueError):configuration('auto',-1)
