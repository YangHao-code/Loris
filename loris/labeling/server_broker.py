"""Server-resident directory broker; credentials never enter experiment artifacts."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from loris._engine.rlearner import atomic_json
from .llm_provider import propose, validate_request


def full_answer(request, truth_path):
    with np.load(truth_path, allow_pickle=False) as archive:
        matches = np.flatnonzero(archive['row_ids'] == request['row_id'])
        if len(matches) != 1 or archive['labels'].tolist() != request['labels']:
            raise ValueError('Requested document or complete label axis differs')
        position = int(matches[0])
        if not archive['observed'][position].all():
            raise ValueError('Human answer must contain every GT label')
        targets = archive['targets'][position].astype(int).tolist()
    return dict(row_id=request['row_id'], labels=request['labels'], targets=targets,
        provider='simulated_human', provenance='Server-local complete GT for this explicitly requested document after LLM fallback')


def process_request(suite, request_path, truth_path, api_key, cache, sources=()):
    suite, request_path, cache = Path(suite), Path(request_path), Path(cache)
    stem, kind, suffix = request_path.name.split('.')
    if not stem.isdigit() or kind not in ('llm', 'human') or suffix != 'json':
        raise ValueError('Invalid external request filename')
    request = validate_request(json.loads(request_path.read_text()))
    axis = json.loads((suite / 'TARGET_AXIS.json').read_text())
    lookup = dict(zip(axis['row_ids'], axis['rows']))
    if request['labels'] != axis['labels'] or lookup.get(request['row_id']) != request['row'] or int(stem) != request['row']:
        raise ValueError('Request is outside the exact declared target axis')
    identity = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    directory = cache / identity
    directory.mkdir(parents=True, exist_ok=True)
    answer_name = stem + '.' + kind + '.json'
    api_name = stem + '.llm.api.json'
    for source in sources:
        original = Path(source) / 'external/requests' / request_path.name
        if original.is_file() and json.loads(original.read_text()) == request:
            for name in [answer_name, api_name]:
                saved = Path(source) / 'external/answers' / name
                target = directory / name
                if saved.is_file() and not target.exists():
                    atomic_json(target, json.loads(saved.read_text()))
    answer = directory / answer_name
    receipt = directory / api_name
    if kind == 'llm':
        if not receipt.exists() and not answer.exists():
            try:
                propose(request, directory, api_key=api_key)
            except RuntimeError:
                if not receipt.exists():
                    raise
        if not receipt.exists():
            raise ValueError('LLM proposal requires an API attempt receipt')
    else:
        prior = suite / 'external/answers' / api_name
        if not prior.exists():
            raise ValueError('Human fallback requires a preceding LLM API attempt')
        if not answer.exists():
            atomic_json(answer, full_answer(request, truth_path))
    for path in [answer, receipt] if kind == 'llm' else [answer]:
        if path.is_file():
            target = suite / 'external/answers' / path.name
            payload = json.loads(path.read_text())
            if target.exists():
                if json.loads(target.read_text()) != payload:
                    raise ValueError('Existing external artifact has another identity')
            else:
                atomic_json(target, payload)
    atomic_json(suite / 'external/ready' / request_path.name,
        dict(row_id=request['row_id'], kind=kind, status='answer_ready' if answer.exists() else 'llm_unavailable'))
    return dict(row_id=request['row_id'], kind=kind, available=answer.exists(), reused=any(Path(source) == suite for source in sources))


def evaluate_suite(suite, truth_path, *, test_used_for_selection=False):
    suite = Path(suite)
    event_path = suite / 'labeling/EVENTS.jsonl'
    if not event_path.exists():
        return dict(status='waiting_for_start', epoch=time.time())
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
    accepted = {event['row_id'] for event in events if event.get('kind') == 'answer_accepted'}
    with np.load(truth_path, allow_pickle=False) as archive:
        rows, labels, truth = archive['row_ids'].tolist(), archive['labels'].tolist(), archive['targets'].astype(bool)
        if not archive['observed'].all():
            raise ValueError('Complete scoring truth required')
    if not accepted <= set(rows):
        raise ValueError('Accepted answers outside evaluation axis')
    from loris.experiments.aux_tuning import scores
    unaccepted = np.asarray([row not in accepted for row in rows], bool)
    phases = ['initial_zero_update']
    for event in events:
        if event.get('kind') == 'query_batch_completed' and event.get('accepted', 0):
            phases.append('answers_only_zero_update')
        elif event.get('kind') == 'model_update_committed':
            phases.append('committed_model_update')
    rounds = []
    for path in sorted((suite / 'labeling/rounds').glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            if archive['row_ids'].tolist() != rows or archive['labels'].tolist() != labels:
                raise ValueError('Prediction and complete evaluation axes differ')
            index = int(path.stem)
            record = dict(snapshot=path.name, phase=phases[index] if index < len(phases) else 'awaiting_commit_event')
            for name, key in [('aux', 'aux_raw'), ('chase', 'output_positive')]:
                record[name] = scores(truth, archive[key])
                record['unaccepted_' + name] = scores(truth[unaccepted], archive[key][unaccepted]) if unaccepted.any() else None
                record['accepted_' + name] = scores(truth[~unaccepted], archive[key][~unaccepted]) if (~unaccepted).any() else None
            rounds.append(record)
    first = next((index for index, record in enumerate(rounds) if record['phase'] == 'committed_model_update'), None)
    comparison = dict(status='waiting_first_committed_model_update')
    if first is not None and first > 0:
        comparison = dict(status='evaluated', before=rounds[first - 1]['snapshot'], after=rounds[first]['snapshot'])
        for name in ['aux', 'chase']:
            before, after = rounds[first - 1]['unaccepted_' + name], rounds[first]['unaccepted_' + name]
            comparison[name] = {metric: after[metric] - before[metric]
                for metric in ['micro_f1', 'macro_f1', 'mean_f1']} if before and after else None
    result = dict(status='evaluated', epoch=time.time(), suite=suite.name,
        execution_location='experiment_server', accepted_documents=len(accepted),
        unaccepted_documents=int(unaccepted.sum()), rounds=rounds, first_update=comparison,
        comparison_mask='same audit-time unaccepted rows in every snapshot',
        test_used_for_hyperparameter_selection=test_used_for_selection,
        score_interpretation='Interactive test evaluation; overall scores include obtained answers',
        training_attempts=sum(event.get('kind') == 'model_training_started' for event in events),
        committed_updates=sum(event.get('kind') == 'model_update_committed' for event in events))
    atomic_json(suite / 'SERVER_EVALUATION.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--api-key-file', required=True)
    parser.add_argument('--state-directory', required=True)
    args = parser.parse_args(argv)
    key_path = Path(args.api_key_file)
    if key_path.stat().st_mode & 0o077:
        raise ValueError('API key file must be owner-only')
    api_key = key_path.read_text().strip()
    state = Path(args.state_directory)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_evaluated = {}
        while not (state / 'STOP').exists():
            configuration = json.loads(Path(args.config).read_text())
            suites = [Path(value) for value in configuration['suites']]
            statuses = {}
            for suite in suites:
                try:
                    pending = [path for path in sorted((suite / 'external/requests').glob('*.json'))
                        if not (suite / 'external/ready' / path.name).exists()]
                    if pending:
                        process_request(suite, pending[0], configuration['truth'], api_key, state / 'cache', sources=suites)
                    event_path = suite / 'labeling/EVENTS.jsonl'
                    stamp = event_path.stat().st_mtime_ns if event_path.exists() else 0
                    if stamp != last_evaluated.get(str(suite)):
                        evaluation = evaluate_suite(suite, configuration['truth'],
                            test_used_for_selection=configuration.get('test_used_for_selection', False))
                        last_evaluated[str(suite)] = stamp
                    else:
                        evaluation_path = suite / 'SERVER_EVALUATION.json'
                        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
                    statuses[suite.name] = dict(status='watching', pending=len(pending),
                        accepted=evaluation.get('accepted_documents', 0),
                        committed_updates=evaluation.get('committed_updates', 0))
                    axis_path = suite / 'TARGET_AXIS.json'
                    if axis_path.exists():
                        axis = json.loads(axis_path.read_text())
                        if hashlib.sha256((suite / 'manifest.json').read_bytes()).hexdigest() != axis['manifest_sha256']:
                            raise ValueError('Broker target belongs to another manifest')
                        atomic_json(suite / 'BROKER_READY.json', dict(status='complete', epoch=time.time(),
                            manifest_sha256=axis['manifest_sha256'], execution_location='experiment_server',
                            credentials_in_experiment_artifacts=False))
                except Exception as error:
                    statuses[suite.name] = dict(status='error', error_type=type(error).__name__)
            atomic_json(state / 'STATUS.json', dict(status='watching', epoch=time.time(), pid=os.getpid(),
                execution_location='experiment_server', suites=statuses))
            time.sleep(3)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
