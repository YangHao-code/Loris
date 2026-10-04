"""Read-only validation of a completed aligned selector's entire artifact chain."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .selection import (SCHEMA, ModelSelectionProtocol, _validate, _normalize_model_interpretation,
                        majority_predictions, ORACLE_POLICIES)
from .patterns import macro_f1
from loris._engine.rlearner import digest, array_identity
from loris._engine.model_selection import _sha_file, _load_selector_checkpoint


def verify_model_selection_output(directory):
    directory = Path(directory).resolve(strict=True)
    files = {}
    def path(relative, expected=None):
        p = directory/relative
        if not p.is_file():
            raise ValueError('missing selector artifact: '+relative)
        actual = _sha_file(p)
        if expected is not None and actual != expected:
            raise ValueError('selector artifact checksum differs: '+relative)
        if str(p) in files and files[str(p)] != actual:
            raise ValueError('selector artifact changed during repeated read: '+relative)
        files[str(p)] = actual
        return p
    def read(relative):
        return json.loads(path(relative).read_text())
    protocol, result, complete = read('PROTOCOL.json'), read('SELECTED.json'), read('COMPLETE.json')
    bid = digest(protocol)
    if (protocol.get('schema') != SCHEMA or result.get('schema') != SCHEMA
            or complete.get('binding_id') != bid or result.get('binding_id') != bid
            or complete.get('result_sha256') != digest(result)
            or read('SELECTION.json') != result):
        raise ValueError('model selector producer protocol/result binding differs')
    if protocol.get('artifact_contract') != 'complete_input_arrays_training_chain_and_model_interpretation_v2':
        raise ValueError('selector lacks authenticated complete input arrays; rebind or rerun explicitly')
    declaration = protocol.get('protocol')
    if not isinstance(declaration, dict) or not set(ModelSelectionProtocol.__dataclass_fields__) <= set(declaration):
        raise ValueError('selector training protocol requires its explicit oracle policy and complete parameters')
    cfg = ModelSelectionProtocol(**{name:declaration[name] for name in ModelSelectionProtocol.__dataclass_fields__})
    if cfg.manifest() != protocol['protocol']:
        raise ValueError('selector training protocol is incomplete or differs')
    if result.get('oracle_policy') != cfg.oracle_policy:
        raise ValueError('selector result oracle policy differs from its training protocol')
    if not isinstance(result.get('input_arrays_sha256'),str):
        raise ValueError('selector complete input array SHA is missing')
    with np.load(path('INPUT_ARRAYS.npz',result['input_arrays_sha256']),allow_pickle=False) as raw:
        if set(raw.files) != {'features','decisions','truth','assignments'}:
            raise ValueError('selector full input array inventory differs')
        data = {key:raw[key] for key in raw.files}
    for key,value in data.items():
        if array_identity(value) != protocol.get(key):
            raise ValueError('selector input array binding differs: '+key)
    x,d,y,a,rows,aliases,labels = _validate(data['features'],data['decisions'],data['truth'],data['assignments'],
        protocol['rows'],protocol['aliases'],protocol['labels'],protocol['input_binding'],cfg)
    for source,sha in protocol.get('source_files',{}).items():
        source_path=Path(source).resolve(strict=True)
        if _sha_file(source_path) != sha:
            raise ValueError('model selector producer source differs: '+source)
        files[str(source_path)]=sha
    if not protocol.get('source_files'):
        raise ValueError('selector producer source inventory is missing')
    interpretation = None
    if protocol['model_interpretation'] is not None:
        meta = protocol['model_interpretation']
        if not isinstance(result.get('model_interpretation_sha256'),str):
            raise ValueError('selector model interpretation SHA is missing')
        with np.load(path('MODEL_INTERPRETATION.npz',result.get('model_interpretation_sha256')),allow_pickle=False) as raw:
            expected_keys = {item['array_key'] for kind in ('probabilities','decisions') for item in meta[kind].values()}
            if set(raw.files) != expected_keys:
                raise ValueError('selector model interpretation array inventory differs')
            interpretation = {'row_ids':meta['rows']}
            for kind in ('probabilities','decisions'):
                interpretation[kind] = {alias:raw[item['array_key']] for alias,item in meta[kind].items()}
        rebuilt,_ = _normalize_model_interpretation(interpretation,aliases,labels,rows,d)
        if rebuilt != meta:
            raise ValueError('selector model interpretation binding differs')
    elif result.get('model_interpretation_sha256') is not None:
        raise ValueError('unbound model interpretation result')
    clusters = list(np.unique(a))
    if len(result.get('clusters',())) != len(clusters):
        raise ValueError('selector omitted or added clusters')
    expected_artifacts=set()
    for pos in range(len(clusters)):
        prefix='cluster_%02d/'%pos
        expected_artifacts.update(prefix+name for name in
            ('INPUTS.json','ORACLE.json','oracle.npz','final_selection.npz','SELECTION.json','COMPLETE.json'))
        for epoch in range(cfg.epochs+1):
            expected_artifacts.update((prefix+'epoch_%05d.json'%epoch,prefix+'checkpoint_%05d.pt'%epoch))
    archived=result.get('training_artifact_sha256')
    if not isinstance(archived,dict) or set(archived)!=expected_artifacts:
        raise ValueError('selector full training artifact inventory differs')
    for relative,sha in archived.items():
        if not isinstance(sha,str):
            raise ValueError('selector training artifact SHA is missing')
        path(relative,sha)
    expected_aliases = {}
    for pos,cluster in enumerate(clusters):
        prefix='cluster_%02d/'%pos
        indices=np.flatnonzero(a==cluster)
        inp,sub,done=read(prefix+'INPUTS.json'),read(prefix+'SELECTION.json'),read(prefix+'COMPLETE.json')
        if (inp.get('binding_id') != bid or inp.get('cluster') != int(cluster)
                or inp.get('oracle_policy') != cfg.oracle_policy or sub.get('oracle_policy') != cfg.oracle_policy
                or inp.get('rows') != indices.tolist() or inp.get('row_ids') != [rows[i] for i in indices]
                or result['clusters'][pos] != sub or sub.get('binding_id') != bid
                or done.get('binding_id') != bid or done.get('result_sha256') != digest(sub)
                or sub.get('cluster_id') != int(cluster) or sub.get('rows') != len(indices)
                or sub.get('completed_epochs') != cfg.epochs or sub.get('optimizer_steps') != cfg.epochs):
            raise ValueError('selector cluster input/completion chain differs')
        path(prefix+'oracle.npz',inp.get('oracle_sha256'))
        oracle_json=path(prefix+'ORACLE.json',inp.get('oracle_scores_sha256'))
        oracle=json.loads(oracle_json.read_text())
        if (oracle.get('oracle_policy') != cfg.oracle_policy
                or oracle.get('oracle') != ORACLE_POLICIES[cfg.oracle_policy]):
            raise ValueError('selector oracle policy differs from its training protocol')
        if oracle.get('subsets_evaluated') != math.comb(len(aliases),cfg.k) or sub.get('oracle_subsets_evaluated') != oracle.get('subsets_evaluated'):
            raise ValueError('selector oracle did not cover all K-subsets')
        if sub.get('oracle_best_global_macro_subset') != oracle.get('best_global_macro_subset'):
            raise ValueError('selector oracle summary differs')
        receipts=sorted((directory/prefix).glob('epoch_*.json'))
        if [p.name for p in receipts] != ['epoch_%05d.json'%e for e in range(cfg.epochs+1)]:
            raise ValueError('selector training epoch sequence incomplete')
        final=None
        for epoch in range(cfg.epochs+1):
            rec=read(prefix+'epoch_%05d.json'%epoch)
            expected_name='checkpoint_%05d.pt'%epoch
            cp=path(prefix+expected_name,rec.get('sha256'))
            # Local producer checksum is authenticated before deserialization.
            state=_load_selector_checkpoint(cp,map_location='cpu')
            expected={'binding_id':bid,'epoch':epoch,'checkpoint':expected_name,'sha256':files[str(cp)],
                'rows_used':len(indices),'losses_before_update':state.get('losses_before_update'),'summary':state.get('summary')}
            if rec != expected or state.get('binding_id') != bid or state.get('epoch') != epoch or state.get('cluster') != int(cluster):
                raise ValueError('selector checkpoint/epoch receipt binding differs')
            steps=[v.get('step') for v in state.get('optimizer',{}).get('state',{}).values()]
            if (epoch and (len(steps)!=4 or any(float(step)!=epoch for step in steps))) or (not epoch and steps):
                raise ValueError('selector optimizer step state differs from completed epoch')
            if epoch and (not isinstance(state['losses_before_update'],dict)
                    or not all(np.isfinite(v) for v in state['losses_before_update'].values())):
                raise ValueError('selector completed epoch lacks finite losses')
            if epoch==cfg.epochs:
                final=state
        path(prefix+'checkpoint_%05d.pt'%cfg.epochs,sub.get('final_checkpoint_sha256'))
        with np.load(path(prefix+'final_selection.npz',sub.get('final_arrays_sha256')),allow_pickle=False) as raw:
            if set(raw.files) != {'row_indices','scores','masks','frequency'}:
                raise ValueError('selector final array inventory differs')
            fs={key:raw[key] for key in raw.files}
        masks,scores=fs['masks'],fs['scores']
        if (not np.array_equal(fs['row_indices'],indices) or scores.shape!=(len(indices),len(aliases))
                or not np.isfinite(scores).all() or masks.shape!=scores.shape or not np.isin(masks,[0,1]).all()
                or not np.all(masks.sum(axis=1)==cfg.k) or not np.array_equal(scores,final['scores'])
                or not np.array_equal(masks,final['masks'])):
            raise ValueError('selector final masks/scores/rows differ')
        expected_mask=np.zeros_like(masks)
        np.put_along_axis(expected_mask,np.argsort(-scores,axis=1,kind='stable')[:,:cfg.k],1,axis=1)
        frequency=masks.sum(axis=0,dtype=np.int64)
        chosen=np.argsort(-frequency,kind='stable')[:cfg.k].tolist()
        static=np.zeros_like(masks);static[:,chosen]=1
        summary={'frequency':frequency.tolist(),'selected_indices':chosen,'selected_aliases':[aliases[i] for i in chosen],
            'selected_subset_macro_f1':macro_f1(y[indices],majority_predictions(d[:,indices],static)),
            'per_document_selection_macro_f1':macro_f1(y[indices],majority_predictions(d[:,indices],masks))}
        if (not np.array_equal(expected_mask,masks) or not np.array_equal(frequency,fs['frequency'])
                or summary!=final['summary'] or any(sub.get(key)!=value for key,value in summary.items())):
            raise ValueError('selector final frequency/majority metrics differ')
        expected_aliases[str(int(cluster))]=summary['selected_aliases']
    if (result.get('selected_aliases_by_cluster')!=expected_aliases
            or result.get('completed_optimizer_steps')!=cfg.epochs*len(clusters)
            or result.get('complete_fit_rows')!=len(rows)):
        raise ValueError('selector global frequency/completion summary differs')
    for source,sha in files.items():
        if _sha_file(source) != sha:
            raise ValueError('selector artifact changed during complete validation: '+source)
    return {'protocol':protocol,'result':result,'input_arrays':data,
            'model_interpretation':interpretation,'files':files}
