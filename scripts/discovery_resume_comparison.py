"""Exactly resume existing quality-comparison rules, then seal campaign receipts.

Launch through discovery_fast_resume.py with the ORIGINAL release PYTHONPATH:
  --script discovery_resume_comparison.py -- --config CAMPAIGN/CONFIG.json --task aapd

This never reruns selection, pattern extraction, initialization or old completed
trials. Original run_discovery(..., resume=True) validates the full namespace;
its normal recovery may replay a pending score/callback after interruption.
"""
import argparse
import json
from pathlib import Path


def resume(config_path, task):
    import loris
    from loris._engine.rlearner import immutable_json
    from loris.discovery.manifest import load_inputs, file_record
    from loris.discovery.runner import run_discovery

    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    spec = config['tasks'][task]
    release = Path(config['release']).resolve(strict=True)
    if Path(loris.__file__).resolve().parent != release / 'loris':
        raise ValueError('resume must import the unchanged original release')
    deployed = json.loads((release / 'DEPLOY_RECEIPT.json').read_text())
    if deployed['git_head'] != config['git_head']:
        raise ValueError('original deployment commit differs from campaign')
    code = json.loads((config_path.parent / 'CODE_MANIFEST.json').read_text())
    for relative, expected in code['release_files'].items():
        if file_record(release / relative)['sha256'] != expected:
            raise ValueError('original release file changed: ' + relative)
    output = Path(spec['output']).resolve(strict=True)
    comparison = output / 'rules_main'
    plan = json.loads((comparison / 'PLAN.json').read_text())
    if not plan['execute_BO'] or not plan['execute_selectors']:
        raise ValueError('existing comparison must declare executed selection and BO')
    summaries = []
    for case in plan['cases']:
        folder = comparison / case['name']
        if folder.parent != comparison or len(Path(case['name']).parts) != 1:
            raise ValueError('comparison case path must be a direct child')
        manifest = folder / 'discovery.json'
        inputs = load_inputs(manifest)
        if (inputs['budget_per_cluster'] != plan['budget_per_cluster']
                or inputs['seed'] != plan['rule_seed'] or inputs['workers'] != plan['workers']):
            raise ValueError('resume manifest budget/seed/workers differ from original plan')
        selected = json.loads((folder / 'selector/COMPLETE.json').read_text())
        receipt = run_discovery(output=folder / 'rules', resume=True, **inputs)
        summaries.append(dict(case=case['name'], status='rule_discovery_complete',
                              expert_union=selected['expert_union'], manifest=str(manifest),
                              rule_command=case['rule_command'],
                              rule_receipt=file_record(folder / 'rules/COMPLETE.json'),
                              complete_BO_trials=receipt['complete_BO_trials']))
    result = dict(status='complete', stage='rule_discovery', cases=summaries, output=str(comparison))
    immutable_json(comparison / 'COMPLETE.json', result)
    outer = dict(status='complete', train=spec['train'], dev=spec['dev'], labels=spec['labels'],
                 release=str(release), result=result, test_read=False)
    immutable_json(output / 'RULES_COMPLETE.json', outer)
    return outer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    print(json.dumps(resume(args.config, args.task), indent=2))
