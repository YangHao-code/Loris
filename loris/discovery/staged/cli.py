"""Run staged BO from caller-supplied, checksum-bound train/dev assets."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from loris.discovery.manifest import bound, file_record
from .pipeline import discover


def run(manifest, output):
    path = Path(manifest).resolve(strict=True)
    data = json.loads(path.read_text())
    required = {'schema', 'context', 'pool', 'partitions', 'old_rules', 'settings'}
    if set(data) - required - {'patterns', 'lexical_cache_dir'} or required - set(data):
        raise ValueError('Unknown or missing staged manifest fields')
    if data['schema'] != 'loris_staged_discovery_inputs_v1':
        raise ValueError('Unsupported staged manifest schema')
    refs = {key: bound(data[key], path.parent)
            for key in ('context', 'pool', 'partitions', 'old_rules')}
    if 'patterns' in data:
        refs['patterns'] = bound(data['patterns'], path.parent)
    binding = dict(manifest=file_record(path), inputs={k: file_record(v) for k, v in refs.items()})
    # Context pickle must come from the caller's trusted local preparation.
    # The checksum binds identity, not the safety of a downloaded pickle.
    with refs['context'].open('rb') as stream:
        context = pickle.load(stream)
    pool = json.loads(refs['pool'].read_text())
    rules = json.loads(refs['old_rules'].read_text())['rules']
    patterns = json.loads(refs['patterns'].read_text()) if 'patterns' in refs else None
    with np.load(refs['partitions'], allow_pickle=False) as a:
        if a['labels'].tolist() != list(context['labels']) or a['row_ids'].tolist() != list(context['row_ids']):
            raise ValueError('Partition/context label or document order differs')
        cache = data.get('lexical_cache_dir')
        result = discover(context=context, pool=pool, old_rules=rules,
            A=a['A'], truth_A=a['truth_A'], B=a['B'], truth_B=a['truth_B'],
            output=output, pattern_supply=patterns,
            lexical_cache_dir=(path.parent / cache).resolve() if cache else None,
            **data['settings'])
    after = dict(manifest=file_record(path), inputs={k: file_record(v) for k, v in refs.items()})
    if after != binding:
        (Path(output) / 'COMPLETE.json').unlink()
        (Path(output) / 'FAILED.json').write_text(json.dumps(dict(reason='Input changed during discovery')))
        raise RuntimeError('Inputs changed during discovery; completion invalidated')
    (Path(output) / 'INPUTS.json').write_text(json.dumps(binding, indent=2))
    return dict(selected=result['selected_candidate_id'], output=str(Path(output).resolve()),
                native_verified=True, B_used_for_pruning=result['settings']['B_used_for_pruning'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.manifest, args.output), indent=2))
    return 0
