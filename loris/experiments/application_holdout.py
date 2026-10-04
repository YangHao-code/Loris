"""Explicit application/calibration partition; full official dataset stays audited.

This optional interface supplements the experiment contract, not Chase semantics.
Only the exact independently audited derived train rule_q can be held out; the original
Gamma must survive unchanged by document ID and all test documents remain.
"""
import json
from pathlib import Path
import pickle
import numpy as np
from loris.discovery.manifest import bound, file_record


def validate_holdout(record, base, context, source_splits, labels):
    path = bound(record, base)
    data = json.loads(path.read_text())
    if set(data) != {'schema', 'role_manifest', 'rule_q_manifest', 'calibration_protocol', 'source_context'} or data['schema'] != 'loris_application_calibration_holdout_v2':
        raise ValueError('Explicit original roles, independently audited rule_q, q protocol and source Gamma required')
    refs = {k: bound(data[k], path.parent) for k in ('role_manifest','rule_q_manifest','calibration_protocol','source_context')}
    roles = json.loads(refs['role_manifest'].read_text())
    if roles.get('status') != 'complete' or roles.get('labels') != labels:
        raise ValueError('Frozen complete role manifest/label axis required')
    qrole = json.loads(refs['rule_q_manifest'].read_text())
    if qrole.get('schema') != 'loris_derived_rule_q_role_v1' or qrole.get('status') != 'complete_independence_audited' or qrole.get('labels') != labels:
        raise ValueError('Independently audited derived rule_q role required; mapping calibration is forbidden')
    if qrole['parent_role_manifest'] != file_record(refs['role_manifest']):
        raise ValueError('Derived rule_q belongs to another original role bundle')
    exposure_path = bound(qrole['exposure_manifest'], refs['rule_q_manifest'].parent)
    selection_path = bound(qrole['selection'], refs['rule_q_manifest'].parent)
    exposure = json.loads(exposure_path.read_text()); selection = json.loads(selection_path.read_text())
    if exposure.get('status') != 'complete_actual_history_audit' or exposure.get('labels') != labels or len(set(exposure.get('model_aliases', []))) != 16:
        raise ValueError('All16 actual training/selection/mapping exposure inventory required')
    paths = {k: bound(roles['roles'][k]['npz'], refs['role_manifest'].parent)
             for k in ('calibration','source_fit','model_tune','selector_fit')}
    paths['rule_q'] = bound(qrole['npz'], refs['rule_q_manifest'].parent)
    axes = {}
    for role, p in paths.items():
        with np.load(p, allow_pickle=False) as z:
            rows = z['row_ids'].tolist(); groups = z['group_ids'].tolist()
            if z['labels'].tolist() != labels or len(set(rows)) != len(rows):
                raise ValueError('Role document or label axis differs')
        count = qrole['rows'] if role == 'rule_q' else roles['roles'][role]['rows']
        if len(rows) != count or len(rows) != len(groups):
            raise ValueError('Frozen role row count differs')
        axes[role] = (rows, set(groups))
    held = axes['rule_q'][0]; holdset = set(held)
    if not held or any(source_splits.get(row) != 'train' for row in held):
        raise ValueError('Only actual official-train rule_q IDs may be excluded; never test/unknown IDs')
    if selection.get('row_ids') != held or set(selection.get('group_ids', [])) != axes['rule_q'][1] or selection.get('truth_used_for_membership') is not False:
        raise ValueError('rule_q membership differs from the truth-free frozen selection')
    if selection.get('exposure_manifest') != file_record(exposure_path):
        raise ValueError('rule_q selection exposure evidence differs')
    entries=exposure.get('exposures', [])
    required={'native7_weight_fit','native7_checkpoint_and_threshold_selection','added9_weight_fit',
        'added9_checkpoint_selection','all16_probability_mapping_and_threshold_fit',
        'selector_fit_rule_fit_and_Gamma','rule_BO_selection'}
    if {x.get('kind') for x in entries} != required:
        raise ValueError('Exposure inventory omits actual supervised/calibration stages')
    exposed_rows={r for x in entries for r in x['row_ids']}
    exposed_groups={g for x in entries for g in x['group_ids']}
    if set(exposure['exposed_row_ids']) != exposed_rows or set(exposure['exposed_group_ids']) != exposed_groups:
        raise ValueError('Exposure inventory union is incomplete')
    if holdset & exposed_rows or axes['rule_q'][1] & exposed_groups:
        raise ValueError('rule_q overlaps actual historical training/selection/calibration rows or groups')
    for role in ('source_fit','model_tune','calibration','selector_fit'):
        if not set(axes[role][0]) <= exposed_rows or not axes[role][1] <= exposed_groups:
            raise ValueError('Exposure inventory omits a frozen supervised role: ' + role)
    if axes['rule_q'][1] & set(exposure.get('test_text_duplicate_groups_excluded', [])):
        raise ValueError('rule_q overlaps complete-test text groups')
    protocol = json.loads(refs['calibration_protocol'].read_text())
    from loris.labeling.reliability import validate_protocol
    validate_protocol(protocol, context=context)
    if protocol['calibration_row_ids'] != held:
        raise ValueError('Excluded axis must exactly equal independent rule_q and q protocol')
    model_rows={r for x in entries if x['kind'] not in ('selector_fit_rule_fit_and_Gamma','rule_BO_selection') for r in x['row_ids']}
    rule_rows={r for x in entries if x['kind'] in ('selector_fit_rule_fit_and_Gamma','rule_BO_selection') for r in x['row_ids']}
    if not model_rows <= set(protocol['model_fit_row_ids']):
        raise ValueError('q protocol omits model fitting, native dev selection or mapping/threshold exposure')
    if not rule_rows <= set(protocol['rule_fit_row_ids']):
        raise ValueError('q protocol omits actual rule-fit/BO selection exposure')
    with refs['source_context'].open('rb') as stream:
        original = pickle.load(stream)
    if original['labels'] != labels or len(set(original['row_ids'])) != len(original['row_ids']):
        raise ValueError('Original discovery Gamma axis differs')
    old = {row:i for i,row in enumerate(original['row_ids'])}
    expected_pos = np.zeros_like(context['gamma'].pos)
    expected_neg = np.zeros_like(context['gamma'].neg)
    now = {row:i for i,row in enumerate(context['row_ids'])}
    for row, index in old.items():
        signed = original['gamma'].pos[index] | original['gamma'].neg[index]
        if signed.any() and (row in holdset or row not in now):
            raise ValueError('Original Gamma root may not be held out or omitted')
        if row in now:
            expected_pos[now[row]] = original['gamma'].pos[index]
            expected_neg[now[row]] = original['gamma'].neg[index]
    if not np.array_equal(expected_pos, context['gamma'].pos) or not np.array_equal(expected_neg, context['gamma'].neg):
        raise ValueError('Application Gamma differs from unchanged original signed roots')
    for name in ('subset','strict_subset'):
        expected = set()
        for a,b in getattr(original['gamma'],name,()):
            if original['row_ids'][a] not in now or original['row_ids'][b] not in now:
                raise ValueError('Original Gamma relationship cannot be omitted')
            expected.add((now[original['row_ids'][a]],now[original['row_ids'][b]]))
        if expected != set(getattr(context['gamma'],name,())):
            raise ValueError('Application Gamma relationships changed')
    files = [file_record(path), file_record(exposure_path), file_record(selection_path)] + [file_record(p) for p in list(refs.values()) + list(paths.values())]
    return holdset, dict(status='verified_independent_train_rule_q_holdout', rows=len(held),
        role='rule_q', official_test_excluded=0, original_Gamma_unchanged=True,
        experiment_partition_supplement=True, files=files)
