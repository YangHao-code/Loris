"""Read-only BO stop-boundary check, including pending/tell/sampler split commits.

For a trustworthy snapshot, SIGSTOP every old discovery writer first, then pass
their PIDs with --pid (repeatable). A reported unsealed RUNNING trial must be
allowed to finish sealing in the old process; never delete it or resample it.
Without PIDs this is only an advisory live sample, not authorization to kill.
Full original source/input/observation validation still runs during resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def _json(path):
    return json.loads(path.read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _checkpoint(folder, checkpoint):
    completed = checkpoint['completed_trial']
    if checkpoint.get('observation_count', completed + 1) != completed + 1:
        raise ValueError('checkpoint observation count differs')
    states = folder / 'sampler_states/states'
    for key in ('static_sha256', 'state_sha256'):
        if key in checkpoint:
            path = states / (checkpoint[key] + '.pickle')
            if not path.is_file():
                raise ValueError('missing ' + key + ' blob')
            # The static blob may include the entire fit corpus. Full resume
            # checks its SHA; use the small dynamic state for quick stop checks.
            if key == 'state_sha256' and hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint[key]:
                raise ValueError('dynamic RNG checkpoint SHA differs')
    if completed >= 0 and 'observation_prefix_sha256' in checkpoint:
        raw = (folder / 'sampler_states/observations' / ('%05d.json' % completed)).read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint['observation_prefix_sha256']:
            raise ValueError('last observation checkpoint SHA differs')
    return completed


def cluster_state(folder):
    protocol = _json(folder / 'protocol.json')
    protocol_id = _digest(protocol)
    checkpoint = _json(folder / 'sampler.json')
    completed = _checkpoint(folder, checkpoint)
    database = folder / 'optuna.sqlite'
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.execute('BEGIN')
        trials = db.execute('SELECT trial_id,number,state FROM trials ORDER BY number').fetchall()
        complete = [number for _, number, state in trials if state == 'COMPLETE']
        running = [number for _, number, state in trials if state == 'RUNNING']
        if any(state not in ('COMPLETE', 'RUNNING', 'WAITING') for _, _, state in trials):
            raise ValueError('failed/pruned trial is not recoverable under original protocol')
        if complete != list(range(len(complete))):
            raise ValueError('complete trials are not a consecutive prefix')
        pending_path = folder / 'pending.json'
        if not pending_path.exists():
            if running:
                raise ValueError('unsealed RUNNING trial: thaw original worker and retry')
            if completed != len(complete) - 1:
                raise ValueError('sampler checkpoint does not follow completed trials')
            return dict(cluster=folder.name, safe=True, completed=len(complete), phase='between_trials')
        pending = _json(pending_path)
        number = pending['number']
        if pending['protocol_id'] != protocol_id:
            raise ValueError('pending protocol ID differs')
        matching = [(trial_id, state) for trial_id, n, state in trials if n == number]
        if len(matching) != 1:
            raise ValueError('pending trial missing from Optuna')
        trial_id, state = matching[0]
        if any(n != number for n in running):
            raise ValueError('another unsealed RUNNING trial exists')
        attrs = {key: json.loads(value) for key, value in db.execute(
            'SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (trial_id,))}
        if 'paper36_compact_parameters_v1' not in attrs:
            raise ValueError('only compact discovery checkpoints are supported by this checker')
        if attrs['paper36_compact_parameters_v1'] != pending['params']:
            raise ValueError('pending parameters differ from Optuna')
        if _checkpoint(folder, pending['sampler_before_tell']) != number - 1:
            raise ValueError('pending RNG checkpoint number differs')
        receipt = folder / ('trial_%05d.json' % number)
        if receipt.exists():
            record = _json(receipt)
            if any(record.get(key) != value for key, value in pending.items()):
                raise ValueError('trial receipt differs from pending body')
        elif state == 'COMPLETE':
            raise ValueError('COMPLETE trial missing score receipt')
        if state == 'RUNNING':
            if completed != number - 1 or len(complete) != number:
                raise ValueError('RUNNING trial does not follow checkpoint')
        elif state == 'COMPLETE':
            if completed not in (number - 1, number) or len(complete) != number + 1:
                raise ValueError('COMPLETE trial does not follow checkpoint')
            values = db.execute('SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective',
                                (trial_id,)).fetchall()
            if values != [(record['standalone']['score'],)] or attrs.get('rule_body') != record['rule']:
                raise ValueError('COMPLETE Optuna score/body differs from receipt')
        else:
            raise ValueError('pending trial is neither RUNNING nor COMPLETE')
        return dict(cluster=folder.name, safe=True, completed=len(complete), phase=state,
                    pending=number, score_receipt_exists=receipt.exists())
    finally:
        db.close()


def _frozen(pids):
    result = {}
    pending = list(pids)
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        result[pid] = fields[0]
        pending.extend(int(p) for p in (path / 'task' / str(pid) / 'children').read_text().split())
    if any(state not in ('T', 't') for state in result.values()):
        raise ValueError('all supplied writers and descendants must be stopped: ' + str(result))
    return result


def check(output, pids=()):
    output = Path(output)
    frozen = _frozen(pids) if pids else None
    result = []
    for folder in sorted((output / 'learning').glob('cluster_[0-9][0-9]')):
        if not (folder / 'optuna.sqlite').exists():
            continue
        try:
            result.append(cluster_state(folder))
        except Exception as error:
            result.append(dict(cluster=folder.name, safe=False, error=str(error)))
    if frozen is not None and _frozen(pids) != frozen:
        raise ValueError('writer process snapshot changed during check')
    return dict(schema='loris_readonly_sealed_stop_boundary_v1',
                safe=bool(result) and all(r['safe'] for r in result),
                frozen_writer_pids=frozen, live_sample_only=frozen is None,
                full_original_resume_validation_still_required=True, clusters=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pid', action='append', type=int, default=[])
    args = parser.parse_args()
    result = check(args.output, args.pid)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['safe'] else 2)
