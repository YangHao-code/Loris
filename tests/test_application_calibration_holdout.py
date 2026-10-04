import json
import pickle
import numpy as np
import pytest
from loris.document import Document
from loris._engine.rdl_reference import RefGamma
from loris.discovery.manifest import file_record
from loris.experiments.application_holdout import validate_holdout


def dump(path,value):
    path.write_text(json.dumps(value));return file_record(path)


def setup(tmp,bad=None):
    labels=['A'];role_rows={'source_fit':['model'],'model_tune':['tune'],
        'selector_fit':['root'],'calibration':['mapping']}
    roles={'status':'complete','labels':labels,'roles':{}}
    for i,(role,rows) in enumerate(role_rows.items()):
        p=tmp/(role+'.npz');np.savez(p,row_ids=rows,group_ids=[str(i)],labels=labels)
        roles['roles'][role]={'rows':1,'npz':file_record(p)}
    role_ref=dump(tmp/'ROLE.json',roles)
    entries=[('native7_weight_fit',['model'],['0']),('native7_checkpoint_and_threshold_selection',['dev'],['dev']),
      ('added9_weight_fit',['model'],['0']),('added9_checkpoint_selection',['tune'],['1']),
      ('all16_probability_mapping_and_threshold_fit',['mapping'],['3']),
      ('selector_fit_rule_fit_and_Gamma',['root'],['2']),('rule_BO_selection',['dev'],['dev'])]
    exposure=dict(status='complete_actual_history_audit',labels=labels,model_aliases=['m'+str(i) for i in range(16)],
      exposures=[dict(kind=k,row_ids=r,group_ids=g) for k,r,g in entries],exposed_row_ids=['model','tune','dev','mapping','root'],
      exposed_group_ids=['0','1','2','3','dev'],test_text_duplicate_groups_excluded=[])
    if bad=='missing_stage':exposure['exposures'].pop(4)
    if bad=='incomplete_history':exposure['status']='awaiting_history'
    if bad=='only15models':exposure['model_aliases'].pop()
    ef=dump(tmp/'EXPOSURE.json',exposure)
    qgroups=['3' if bad=='mapping_group' else '0' if bad=='model_group' else 'qgroup']
    qids=['cal']
    np.savez(tmp/'rule_q.npz',row_ids=qids,group_ids=qgroups,labels=labels)
    sf=dump(tmp/'SELECTION.json',dict(row_ids=qids,group_ids=qgroups,truth_used_for_membership=False,exposure_manifest=ef))
    qr=dump(tmp/'RULE_Q.json',dict(schema='loris_derived_rule_q_role_v1',status='complete_independence_audited',labels=labels,
      parent_role_manifest=role_ref,exposure_manifest=ef,selection=sf,npz=file_record(tmp/'rule_q.npz'),rows=1))
    old_rows=['root','cal','model','tune','mapping','dev']
    old=dict(labels=labels,row_ids=old_rows,gamma=RefGamma(np.array([[1],[0],[0],[0],[0],[0]],bool),np.zeros((6,1),bool)))
    old_path=tmp/'source.pkl';old_path.write_bytes(pickle.dumps(old))
    app_rows=['root','model','tune','mapping','dev','test']
    context=dict(row_ids=app_rows,labels=labels,docs=[Document(cnt=x) for x in app_rows],
      gamma=RefGamma(np.array([[1],[0],[0],[0],[0],[0]],bool),np.zeros((6,1),bool)))
    model=['model','tune','mapping','dev']
    if bad=='missing_mapping_protocol':model.remove('mapping')
    if bad=='missing_native_dev_protocol':model.remove('dev')
    protocol=dict(schema='loris_independent_rule_calibration_v2',context={},answers={},
      calibration_row_ids=['other'] if bad=='q_axis' else qids,model_fit_row_ids=model,rule_fit_row_ids=['root','dev'],
      model_binding={},independence_statement='Synthetic independently audited rule_q with Gamma anchors')
    pref=dump(tmp/'Q.json',protocol)
    h=dict(schema='loris_application_calibration_holdout_v2',role_manifest=role_ref,rule_q_manifest=qr,
      calibration_protocol=pref,source_context=file_record(old_path))
    if bad=='legacy_unexposed_claim':h['schema']='loris_application_calibration_holdout_v1';h.pop('rule_q_manifest')
    ref=dump(tmp/'HOLDOUT.json',h)
    splits={k:'train' for k in old_rows};splits['dev']='dev';splits['test']='test'
    if bad=='test_exclusion':splits['cal']='test'
    if bad=='unknown':del splits['cal']
    if bad=='gamma_change':
        pos=context['gamma'].pos.copy();pos[-1,0]=True;context['gamma']=RefGamma(pos,context['gamma'].neg)
    if bad=='gamma_omission':context['row_ids'][0]='replacement'
    return ref,context,splits,labels


def test_explicit_holdout_preserves_original_gamma_and_full_test(tmp_path):
    ref,c,s,l=setup(tmp_path);rows,evidence=validate_holdout(ref,tmp_path,c,s,l)
    assert rows=={'cal'} and evidence['original_Gamma_unchanged']
    assert evidence['official_test_excluded']==0 and evidence['role']=='rule_q'


@pytest.mark.parametrize('bad',['test_exclusion','unknown','gamma_change','gamma_omission','model_group',
    'mapping_group','q_axis','missing_stage','incomplete_history','only15models',
    'missing_mapping_protocol','missing_native_dev_protocol','legacy_unexposed_claim'])
def test_invalid_exclusion_or_actual_exposure_rejected(tmp_path,bad):
    ref,c,s,l=setup(tmp_path,bad)
    with pytest.raises(ValueError):validate_holdout(ref,tmp_path,c,s,l)
