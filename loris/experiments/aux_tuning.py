"""Controlled train-only Aux updates with an explicitly declared selection split."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from loris.discovery.manifest import bound, file_record
from loris._engine.rlearner import atomic_json
from loris.labeling.feedback import Batch, incremental_batch
from loris._engine.rdl_reference import RefGamma


METHODS = ('supervised', 'self_train', 'adaptive_self_train')
INITIALIZATIONS = ('m0', 'previous', 'previous_delta')


def scores(truth, probabilities):
    truth = np.asarray(truth, bool)
    predicted = np.asarray(probabilities) >= .5
    if truth.shape != predicted.shape or truth.ndim != 2 or not len(truth):
        raise ValueError('Nonempty matching multilabel evaluation axes required')
    true_positive = (truth & predicted).sum(0)
    denominator = (truth.sum(0) + predicted.sum(0)).astype(float)
    per_label = np.divide(2 * true_positive, denominator,
        out=np.zeros_like(denominator), where=denominator > 0)
    micro = float(2 * true_positive.sum() / max(denominator.sum(), 1))
    macro = float(per_label.mean())
    return dict(micro_f1=micro, macro_f1=macro, mean_f1=(micro + macro) / 2,
        rows=len(truth), labels=truth.shape[1], threshold=.5)


def pseudo_supervision(probabilities, known, *, method, confidence=.95,
                       minimum_confidence=.8, weak_weight=.1):
    probabilities = np.asarray(probabilities, dtype=float)
    known = np.asarray(known, bool)
    if (probabilities.ndim != 2 or probabilities.shape != known.shape
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('Finite probability and known-label axes required')
    if method not in METHODS or not .5 < minimum_confidence <= confidence <= 1:
        raise ValueError('Explicit method and confidence thresholds required')
    if not np.isfinite(weak_weight) or not 0 < weak_weight <= 1:
        raise ValueError('Weak-label weight must lie in (0,1]')
    unknown = ~known
    positive_counts = ((probabilities >= confidence) & unknown).sum(0)
    negative_counts = ((probabilities <= 1 - confidence) & unknown).sum(0)
    thresholds = np.full((2, probabilities.shape[1]), confidence, dtype=float)
    if method == 'adaptive_self_train':
        for polarity, counts in enumerate([negative_counts, positive_counts]):
            maximum = int(counts.max()) if len(counts) else 0
            if maximum:
                progress = counts / maximum
                mapped = confidence * progress / (2 - progress)
                thresholds[polarity] = np.where(counts > 0,
                    np.clip(mapped, minimum_confidence, confidence), confidence)
    positive = probabilities >= thresholds[1]
    negative = probabilities <= 1 - thresholds[0]
    observed = unknown & (positive | negative)
    if method == 'supervised':
        observed[:] = False
    targets = (probabilities >= .5).astype(np.float32)
    weights = observed.astype(np.float32) * weak_weight
    audit = dict(method=method, confidence=confidence,
        minimum_confidence=minimum_confidence, weak_weight=weak_weight,
        thresholds_negative=thresholds[0].tolist(), thresholds_positive=thresholds[1].tolist(),
        confident_positive_counts=positive_counts.tolist(), confident_negative_counts=negative_counts.tolist(),
        weak_positive=int((observed & positive).sum()), weak_negative=int((observed & negative).sum()),
        adaptation='per_label_and_polarity_epoch_external_self_training_not_original_FlexMatch',
        unknown_as_negative=False, pseudo_labels_promoted_to_Gamma=False)
    return targets, observed, weights, audit


def build_batch(probabilities, targets, known, configuration, *, fresh_rows=(), previous=None):
    weak_targets, weak, weights, audit = pseudo_supervision(probabilities, known,
        method=configuration['method'], confidence=configuration.get('confidence', .95),
        minimum_confidence=configuration.get('minimum_confidence', .8),
        weak_weight=configuration.get('weak_weight', .1))
    true = np.asarray(known, bool).copy()
    values = np.where(true, targets, weak_targets).astype(np.float32)
    weights[true] = 1
    observed = true | weak
    values[~observed] = 0
    batch = Batch(values, observed, weights, true, weak, audit)
    if configuration['initialization'] == 'previous_delta':
        gamma = RefGamma(true & (targets == 1), true & (targets == 0))
        batch = incremental_batch(batch, gamma, np.ones(len(true), bool),
            np.asarray(fresh_rows, int), .05, previous)
    batch.audit.update(true_cells=int(batch.true_mask.sum()), weak_cells=int(batch.weak_mask.sum()),
        training_rows=int(batch.observed.any(1).sum()), true_mass=float(batch.weights[batch.true_mask].sum()),
        weak_mass=float(batch.weights[batch.weak_mask].sum()))
    return batch


def planned_arms():
    arms = []
    for method in ('supervised', 'self_train'):
        for epochs in (1, 3, 5):
            arms.append(dict(id='%s_m0_e%d' % (method, epochs), method=method,
                initialization='m0', epochs=epochs, rounds=1))
    arms.extend([
        dict(id='adaptive_m0_e3', method='adaptive_self_train', initialization='m0', epochs=3, rounds=1),
        dict(id='self_train_tau90_e3', method='self_train', initialization='m0', epochs=3, rounds=1, confidence=.9),
        dict(id='self_train_weight30_e3', method='self_train', initialization='m0', epochs=3, rounds=1, weak_weight=.3),
        dict(id='self_train_lr10_e3', method='self_train', initialization='m0', epochs=3, rounds=1,
             learning_rate=1e-5, head_learning_rate=5e-5),
    ])
    for initialization in INITIALIZATIONS:
        arms.append(dict(id='self_train_%s_e3_r3' % initialization, method='self_train',
            initialization=initialization, epochs=3, rounds=3))
    return arms


def load_assets(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'loris_train_only_aux_tuning_v1':
        raise ValueError('Audited train-only Aux tuning inputs required')
    files = {name: bound(record, path.parent) for name, record in manifest['files'].items()}
    records = [json.loads(line) for line in files['texts'].read_text().splitlines()]
    with np.load(files['training'], allow_pickle=False) as archive:
        training = {name: archive[name].copy() for name in archive.files}
    with np.load(files['tuning'], allow_pickle=False) as archive:
        tuning = {name: archive[name].copy() for name in archive.files}
    with np.load(files['queries'], allow_pickle=False) as archive:
        queries = {name: archive[name].copy() for name in archive.files}
    identifiers = [record['row_id'] for record in records]
    training_ids = training['row_ids'].tolist()
    tuning_ids = tuning['row_ids'].tolist()
    audit_ids = manifest['audit_row_ids']
    selection_role = manifest.get('selection_role', 'unexposed_train_validation')
    if selection_role not in ('unexposed_train_validation', 'official_test_selection'):
        raise ValueError('Explicit supported model-selection role required')
    selection_marker = ':test:' if selection_role == 'official_test_selection' else ':train:'
    forbidden = set(tuning_ids) | set(audit_ids) | set(manifest['rule_q_row_ids'])
    if (len(identifiers) != len(set(identifiers)) or training_ids + tuning_ids != identifiers
            or set(training_ids) & forbidden or set(tuning_ids) & set(audit_ids)
            or any(':train:' not in row for row in training_ids + audit_ids)
            or any(selection_marker not in row for row in tuning_ids)):
        raise ValueError('Training and selection IDs must match disjoint declared split roles')
    labels = manifest['labels']
    for data in [training, tuning, queries]:
        if data['labels'].tolist() != labels:
            raise ValueError('Full label axis differs')
    if not set(queries['row_ids'].tolist()) <= set(training_ids):
        raise ValueError('Queries must be confined to the training pool')
    if training['targets'].shape != training['observed'].shape or training['targets'].shape != (len(training_ids), len(labels)):
        raise ValueError('Training supervision axes differ')
    if np.any(training['targets'][~training['observed'].astype(bool)] != 0):
        raise ValueError('Unqueried labels must be masked storage placeholders')
    return manifest, files, records, training, tuning, queries


def checkpoint_predictions(checkpoint, manifest, texts):
    from loris.labeling.trainer import load_backend
    implementation = load_backend(manifest['backend_source'])
    import torch
    from loris._engine.model import state_dict_sha256
    configuration = implementation.ChaseBackendConfig(**manifest['configuration'])
    model = implementation.ChaseTransformerBackend(configuration, len(manifest['labels']), device='cuda')
    model._device = model._resolve_device()
    model._tokenizer = model._tokenizer_factory(configuration)
    model._model = model._model_factory(configuration, len(manifest['labels']))
    payload = torch.load(str(checkpoint), map_location='cpu')
    if list(payload['label_names']) != manifest['labels'] or payload['model_name'] != configuration.model_name:
        raise ValueError('Original checkpoint architecture or labels differ')
    model._model.load_state_dict(payload['state_dict'], strict=True)
    expected = state_dict_sha256(payload['state_dict'])
    if state_dict_sha256(model._model.state_dict()) != expected:
        raise ValueError('Baseline parameters differ from original M0')
    del payload
    model._model.to(model._device).eval().requires_grad_(False)
    model._fitted = True
    return model.predict_proba(texts)


def baseline(manifest_path, output):
    manifest, files, records, training, tuning, queries = load_assets(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'STARTED.json', dict(status='running', epoch=time.time()))
    probabilities = checkpoint_predictions(files['checkpoint'], manifest, [record['text'] for record in records])
    np.savez_compressed(output / 'PROBABILITIES.npz', probabilities=probabilities)
    result = dict(status='complete', checkpoint=file_record(files['checkpoint']),
        manifest=file_record(manifest_path), probabilities=file_record(output / 'PROBABILITIES.npz'),
        baseline_tuning=scores(tuning['targets'], probabilities[len(training['row_ids']):]),
        selection_role=manifest.get('selection_role', 'unexposed_train_validation'),
        test_truth_read=manifest.get('selection_role') == 'official_test_selection',
        test_labels_in_gradients=False, audit_truth_read=False)
    atomic_json(output / 'COMPLETE.json', result)
    return result


def run(manifest_path, baseline_path, arm, output):
    manifest, files, records, training, tuning, queries = load_assets(manifest_path)
    if arm['method'] not in METHODS or arm['initialization'] not in INITIALIZATIONS:
        raise ValueError('Unknown training arm')
    if type(arm['epochs']) is not int or arm['epochs'] not in (1, 3, 5) or arm['rounds'] not in (1, 3):
        raise ValueError('Explicit preregistered epoch/round budget required')
    baseline_record = json.loads(Path(baseline_path).read_text())
    if baseline_record['manifest'] != file_record(manifest_path):
        raise ValueError('Baseline belongs to other inputs')
    probability_path = bound(baseline_record['probabilities'], Path(baseline_path).parent)
    with np.load(probability_path, allow_pickle=False) as archive:
        initial_probabilities = archive['probabilities'].copy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'STARTED.json', dict(status='running', epoch=time.time(), arm=arm,
        manifest=file_record(manifest_path), baseline=file_record(baseline_path)))
    from loris.labeling.trainer import train_actual
    training_count = len(training['row_ids'])
    lookup = {row: index for index, row in enumerate(training['row_ids'].tolist())}
    query_rows = [lookup[row] for row in queries['row_ids'].tolist()]
    batch_size = max(32, int(np.ceil(.01 * (~training['observed'].astype(bool).all(1)).sum())))
    targets = np.zeros_like(initial_probabilities, dtype=np.float32)
    known = np.zeros_like(initial_probabilities, dtype=bool)
    targets[:training_count] = training['targets']
    known[:training_count] = training['observed']
    current = initial_probabilities.copy()
    checkpoint = file_record(files['checkpoint'])
    previous_feedback = None
    reports = []
    try:
        for round_index in range(arm['rounds']):
            start = round_index * batch_size
            fresh_rows = query_rows[start:start + batch_size]
            if len(fresh_rows) != batch_size:
                raise ValueError('Shared query ledger is smaller than the registered budget')
            targets[fresh_rows] = queries['targets'][start:start + batch_size]
            known[fresh_rows] = True
            batch = build_batch(current[:training_count], targets[:training_count], known[:training_count],
                arm, fresh_rows=fresh_rows, previous=previous_feedback)
            arrays = {}
            for name in ['targets', 'observed', 'weights', 'true_mask', 'weak_mask']:
                values = getattr(batch, name)
                arrays[name] = np.concatenate([values, np.zeros((len(tuning['row_ids']), values.shape[1]), dtype=values.dtype)])
            full_batch = Batch(**arrays, audit=batch.audit, feedback_state=batch.feedback_state)
            configuration = copy.deepcopy(manifest['configuration'])
            configuration.update(epochs=arm['epochs'], min_epochs=arm['epochs'], patience=0,
                learning_rate=arm.get('learning_rate', 3e-6),
                head_learning_rate=arm.get('head_learning_rate', 1.5e-5))
            parent = file_record(files['checkpoint']) if arm['initialization'] == 'm0' else checkpoint
            round_output = output / ('round_%02d' % (round_index + 1))
            atomic_json(output / 'PROGRESS.json', dict(status='training', round=round_index + 1,
                epoch=time.time(), parent=parent, supervision=batch.audit,
                acquired_training_documents=(round_index + 1) * batch_size))
            current, receipt = train_actual(texts=[record['text'] for record in records],
                labels=manifest['labels'], batch=full_batch, checkpoint=parent['path'],
                checkpoint_sha256=parent['sha256'], source=manifest['backend_source'],
                configuration=configuration, device='cuda', output=round_output,
                gpu_lock=str(output / 'GPU.lock'), max_seconds=21600)
            checkpoint = receipt['checkpoint']
            previous_feedback = batch.feedback_state
            measured = scores(tuning['targets'], current[training_count:])
            report = dict(round=round_index + 1, tuning=measured, parent=parent,
                checkpoint=checkpoint, optimizer_updates=receipt['optimizer_updates'],
                fit_seconds=receipt['model_fit_seconds'], supervision=batch.audit,
                acquired_training_documents=(round_index + 1) * batch_size,
                delta_from_m0={name: measured[name] - baseline_record['baseline_tuning'][name]
                    for name in ['micro_f1', 'macro_f1', 'mean_f1']})
            reports.append(report)
            atomic_json(output / ('ROUND_%02d.json' % (round_index + 1)), report)
        result = dict(status='complete', arm=arm, rounds=reports, baseline=baseline_record['baseline_tuning'],
            final_tuning=reports[-1]['tuning'],
            selection_role=manifest.get('selection_role', 'unexposed_train_validation'),
            test_truth_read=manifest.get('selection_role') == 'official_test_selection',
            test_labels_in_gradients=False, audit_truth_read=False,
            protocol='controlled_train_only_offline_self_training_not_full_Chase',
            query_selection='same_fixed_truth_free_hashed_roster_all_arms',
            unknown_as_negative=False, final_checkpoint=checkpoint)
        atomic_json(output / 'COMPLETE.json', result)
        return result
    except BaseException as error:
        atomic_json(output / 'FAILED.json', dict(status='failed', error_type=type(error).__name__,
            completed_rounds=len(reports), epoch=time.time()))
        raise


def collect(root):
    root = Path(root)
    matrix = json.loads((root / 'MATRIX.json').read_text())
    rows = []
    for arm in matrix['arms']:
        folder = root / 'runs' / arm['id']
        complete = folder / 'COMPLETE.json'
        if complete.is_file():
            result = json.loads(complete.read_text())
            rows.append(dict(id=arm['id'], status='complete', configuration=arm,
                score=result['final_tuning'], delta=result['rounds'][-1]['delta_from_m0']))
        else:
            rows.append(dict(id=arm['id'], configuration=arm,
                status='failed' if (folder / 'FAILED.json').exists() else 'running' if (folder / 'STARTED.json').exists() else 'pending'))
    complete = [row for row in rows if row['status'] == 'complete']
    ordered = sorted(complete, key=lambda row: (-row['score']['mean_f1'], row['id']))
    result = dict(status='complete' if all(row['status'] in ('complete', 'failed') for row in rows) else 'running',
        epoch=time.time(), arms=rows, provisional_best=ordered[0]['id'] if ordered else None,
        selection=matrix['primary_metric'] + '; all_results_including_negative_retained',
        positive_tuning_arms=[row['id'] for row in ordered if row['delta']['mean_f1'] > 0],
        confirmation_status='not_independent_test_when_test_selects_parameters' if matrix.get('selection_role') == 'official_test_selection' else 'held_out_until_search_complete',
        test_used_for_selection=matrix.get('selection_role') == 'official_test_selection')
    atomic_json(root / 'SUMMARY.json', result)
    return result


def confirm(root):
    root = Path(root)
    summary = collect(root)
    if summary['status'] != 'complete' or not summary['provisional_best']:
        raise ValueError('All registered arms must finish or report failure before confirmation')
    output = root / 'confirmation'
    output.mkdir(exist_ok=False)
    winner = summary['provisional_best']
    atomic_json(output / 'SELECTION_FROZEN.json', dict(arm=winner, selection=summary['selection'],
        summary=file_record(root / 'SUMMARY.json'), audit_truth_read=False))
    manifest, files, records, training, tuning, queries = load_assets(root / 'inputs/MANIFEST.json')
    selected = json.loads((root / 'runs' / winner / 'COMPLETE.json').read_text())
    checkpoint = bound(selected['final_checkpoint'], root / 'runs' / winner)
    documents = [json.loads(line) for line in files['audit_texts'].read_text().splitlines()]
    with np.load(files['audit_targets'], allow_pickle=False) as archive:
        if archive['row_ids'].tolist() != [record['row_id'] for record in documents] or archive['row_ids'].tolist() != manifest['audit_row_ids']:
            raise ValueError('Held-out confirmation document axes differ')
        truth = archive['targets'].copy()
        groups = archive['group_ids'].copy()
    texts = [record['text'] for record in documents]
    before = checkpoint_predictions(files['checkpoint'], manifest, texts)
    after = checkpoint_predictions(checkpoint, manifest, texts)
    measured_before, measured_after = scores(truth, before), scores(truth, after)
    unique_groups = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique_groups]
    random = np.random.RandomState(42)
    differences = []
    for repeat in range(500):
        indices = np.concatenate([members[index] for index in random.randint(len(members), size=len(members))])
        differences.append(scores(truth[indices], after[indices])['mean_f1'] - scores(truth[indices], before[indices])['mean_f1'])
    interval = np.percentile(differences, [2.5, 97.5]).tolist()
    delta = {name: measured_after[name] - measured_before[name] for name in ['micro_f1', 'macro_f1', 'mean_f1']}
    np.savez_compressed(output / 'PREDICTIONS.npz', before=before, after=after,
        row_ids=np.asarray(manifest['audit_row_ids']), labels=np.asarray(manifest['labels']))
    result = dict(status='complete', selected_arm=winner, baseline=measured_before,
        selected=measured_after, delta=delta, paired_group_bootstrap_mean_f1_interval=interval,
        bootstrap_replicates=500, positive_point_estimate=delta['mean_f1'] > 0,
        interval_excludes_zero_on_positive_side=interval[0] > 0,
        test_truth_read=False, confirmation_used_to_choose_another_arm=False,
        scope='One held-out confirmation after train-only hyperparameter search')
    atomic_json(output / 'COMPLETE.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    baseline_parser = commands.add_parser('baseline')
    baseline_parser.add_argument('--manifest', required=True)
    baseline_parser.add_argument('--output', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--manifest', required=True)
    run_parser.add_argument('--baseline', required=True)
    run_parser.add_argument('--arm', required=True)
    run_parser.add_argument('--output', required=True)
    collect_parser = commands.add_parser('collect')
    collect_parser.add_argument('--root', required=True)
    confirm_parser = commands.add_parser('confirm')
    confirm_parser.add_argument('--root', required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == 'baseline':
        result = baseline(arguments.manifest, arguments.output)
    elif arguments.command == 'run':
        result = run(arguments.manifest, arguments.baseline,
            json.loads(Path(arguments.arm).read_text()), arguments.output)
    elif arguments.command == 'confirm':
        result = confirm(arguments.root)
    else:
        result = collect(arguments.root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
