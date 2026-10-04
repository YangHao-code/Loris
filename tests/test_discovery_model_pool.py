"""Real small models on synthetic data; no production labels or training."""
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from loris.discovery.datasets import prepare_dataset,SOURCE_SCHEMA
from loris.discovery.manifest import file_record
from loris.discovery.model_pool import (FixedAxisTFIDF,PoolSettings,freeze_pool,run_pool,
    model_configs,MODELS,metrics)
from loris.discovery.supervision import prepare_supervision,load_supervision

def optional_backend_source():
    declared=os.environ.get('LORIS_SSL_SOURCE')
    if not declared:
        pytest.skip('optional seven-model integration: set LORIS_SSL_SOURCE to the real loris_ssl source root')
    backend=Path(declared).expanduser().resolve()
    if not (backend/'loris_ssl/chase_backend.py').is_file():
        pytest.skip('optional seven-model integration: declared LORIS_SSL_SOURCE lacks loris_ssl/chase_backend.py: '+str(backend))
    return backend


def test_fixed_axis_real_svm_and_lr_constants_and_onehot_fit():
    texts=['red apple fruit','red cherry fruit','blue sky cloud','blue ocean water','green forest leaf','green tree leaf']
    target=np.asarray([[0,1,i<3] for i in range(6)],np.int32)
    for method in ('svm','logistic_regression'):
        m=FixedAxisTFIDF(num_labels=3,classifier_type=method,max_features=50,ngram_range=[1,1],min_df=1,
            C=1.,use_calibration=False,max_iter=2000,n_jobs=1).fit(texts,target)
        p=m.predict_proba(texts)
        assert p.shape==(6,3) and (p[:,0]==0).all() and (p[:,1]==1).all()
        assert ((p[:,2]>0)&(p[:,2]<1)).all()
    m=FixedAxisTFIDF(num_labels=3,classifier_type='logistic_regression',max_features=50,ngram_range=[1,1],min_df=1,
        C=1.,use_calibration=False,max_iter=2000,n_jobs=1).fit(texts,np.asarray([[1,0,0]]*3+[[0,1,0]]*3))
    assert m.predict_proba(texts).shape==(6,3) and (m.predict_proba(texts)[:,2]==0).all()
    assert 'unknownword' not in m.vectorizer.vocabulary_


@pytest.mark.parametrize('settings',[PoolSettings(seed=-1),PoolSettings(device='auto'),
    PoolSettings(neural_epochs=2,neural_min_epochs=3),PoolSettings(cpu_threads=0)])
def test_pool_settings_fail_before_execution(settings):
    with pytest.raises(ValueError):settings.validate()


@pytest.fixture(scope='module')
def real_tiny_pool(tmp_path_factory):
    backend=optional_backend_source()
    pytest.importorskip('torch',reason='optional seven-model integration requires torch')
    pytest.importorskip('transformers',reason='optional seven-model integration requires transformers')
    from transformers import BertConfig,BertModel,BertTokenizerFast
    root=tmp_path_factory.mktemp('real_seven_model_fixture');refs={}
    for split,count in [('train',8),('dev',2),('test',1)]:
        path=root/(split+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['text','a','b'])
            for i in range(count):writer.writerow(['red apple fruit' if i%2 else 'blue sky water',int(i%3!=0),int(i%2==0)])
        refs[split]=file_record(path)
    spec=dict(schema=SOURCE_SCHEMA,dataset='tiny_pool',version='fixture',sources=refs,
        columns=dict(text=['text'],id=None,labels=['a','b'],metadata=[]),
        label_axis=dict(mode='declared',labels=['a','b'],provenance='synthetic explicit axis'),
        split=dict(seed=42,fractions=[.1,.1,.8],dev_fraction=.1,group_column=None),
        provenance=dict(source='synthetic only, never an empirical dataset result'))
    prepare_dataset(spec,root/'prepared')
    prepare_supervision(root/'prepared/MANIFEST.json',root/'supervision',historical_exposure={'fixture':'none'},
        fraction_numerator=1,fraction_denominator=2)
    # Real runtime cannot reopen hidden train or any test answers/text.
    for file in ('truth/train.npz','truth/test.npz','documents/test.jsonl'):(root/'prepared'/file).unlink()
    pretrained=root/'tiny_bert';pretrained.mkdir()
    (pretrained/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\napple\nfruit\nblue\nsky\nwater\n')
    BertTokenizerFast(vocab_file=str(pretrained/'vocab.txt')).save_pretrained(pretrained)
    BertModel(BertConfig(vocab_size=11,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,max_position_embeddings=512)).save_pretrained(pretrained)
    settings=PoolSettings(device='cpu',cpu_threads=1,neural_epochs=1,neural_min_epochs=1,neural_patience=1,
        encoder_epochs=1,encoder_min_epochs=1,encoder_patience=1,aux_epochs=1,aux_min_epochs=1,aux_patience=1,
        max_phase_seconds=300)
    frozen=freeze_pool(root/'supervision/SUPERVISION_MANIFEST.json',root/'pool',pretrained_model=pretrained,
        backend_source=backend,settings=settings,gpu_lock=root/'GPU.lock')
    return root,settings,frozen


@pytest.mark.integration
def test_real_seven_model_training_complete_axes_and_bce_checkpoint(real_tiny_pool):
    root,settings,frozen=real_tiny_pool
    assert frozen['status']=='frozen_not_trained' and frozen['new_training']==0
    result=run_pool(root/'pool',phase='all')
    assert result['status']=='complete' and result['shape']==[10,2] and len(result['channels'])==7
    assert result['model_order']==['loris_clf_'+name for name in MODELS]
    data=load_supervision(root/'supervision/SUPERVISION_MANIFEST.json')
    assert len(data['fit_indices'])==4 and result['row_ids']==data['row_ids']
    for name in MODELS:
        record=json.loads((root/'pool'/name/'COMPLETE.json').read_text())
        assert record['shape']==[10,2] and not record['test_read'] and not record['unknown_truth_read']
        with np.load(record['scores']['path'],allow_pickle=False) as z:
            assert z['labels'].tolist()==['a','b'] and z['row_ids'].tolist()==data['row_ids']
            assert np.array_equal(z['global_rows'],np.arange(10))
            assert metrics(data['dev_targets'],z['scores'][data['receiver_indices']])==record['dev_metrics']
        assert record['training_seconds']>=0 and record['inference_seconds']>=0
    for name in ('textcnn','bilstm','encoder_mlp'):
        audit=json.loads((root/'pool'/name/'TRAINING.json').read_text())
        assert audit['selection']=='minimum_dev_BCE_logits' and audit['successful_updates']==1
        assert audit['epochs_run']==1 and audit['fit_rows']==4 and audit['fit_cells']==8
    aux=json.loads((root/'pool/aux_dynamic/TRAINING.json').read_text())
    assert aux['history'] and aux['history'][-1]['optimizer_updates']==1
    before=(root/'pool/HANDOFF.json').read_bytes()
    again=run_pool(root/'pool',phase='cpu')
    assert not again['trained'] and len(again['reused'])==3
    assert (root/'pool/HANDOFF.json').read_bytes()==before


@pytest.mark.integration
def test_frozen_pool_defaults_and_reject_source_change(real_tiny_pool):
    root,settings,_=real_tiny_pool
    p=json.loads((root/'pool/PROTOCOL.json').read_text())
    assert p['models']['encoder_mlp']['num_labels']==2
    assert p['models']['aux_dynamic']['initial_checkpoint_path']==''
    assert p['models']['aux_dynamic']['loss']=='masked_weighted_bce'
    assert 'dev BCE' in p['checkpoint_selection']['native_neural']
    config_file=root/'tiny_bert/config.json';original=config_file.read_bytes()
    config_file.write_bytes(original+b' ')
    try:
        with pytest.raises(ValueError,match='identity differs'):run_pool(root/'pool',phase='handoff')
    finally:config_file.write_bytes(original)
