"""Continue admission from completed BO without rerunning or editing the source.

This creates a new experiment branch with a declared approximate admission
policy. Completed strict prefixes can be preserved; no old F1 score cache or
sampler state is imported. Data, candidates and execution semantics are checked
once at the boundary, not repeatedly hashed inside the candidate loop.
"""
import argparse
import fcntl
import json
import pickle
from pathlib import Path
import time

from loris._engine.rlearner import atomic_json, digest, immutable_json
from loris.rule_forms import parse_author_rule, validate_program_rule_forms
from .runner import RunOptions, _run_identity, evaluator_pair, prove, validate_inputs
from .search.admission import order_trials
from .search.parallel_admission import ParallelScorer, admit_batches
from .search.source_cache import cached_source_extraction
from loris._storage.selection_evaluator import semantic_result_projection
from .manifest import file_record

SCHEMA = "loris_completed_BO_admission_continuation_v1"
DATA_KEYS = ('cluster_order','Si','Pi','relations','row_ids','labels','row_roles',
             'fit_indices','receiver_indices','gamma','truth','probabilities','decisions',
             'document_text_metadata_sha256','context_id','attributes_sha256')


def load_frozen_inputs(manifest, source, *, workers=8, batch_size=8):
    """Authenticate the original manifest and reuse its already frozen Pi/Si.

    Admission continuation does not rerun model/pattern selection. In particular
    it must not demand that an old selector's source *path* equals the current
    release path. The manifest and all bound source files still have to match
    the original BO input record, and continue_admission verifies all runtime
    arrays, primitives and engine semantics against that record before scoring.
    Fresh discovery retains its stronger upstream execution checks unchanged.
    """
    from .manifest import load_manifest
    from loris._engine.rlearner_config import PatternSpec, ParameterDomain
    from loris.predicates import predicate_from_dict
    from loris._engine.rdl_reference import atom_from_dict
    old=json.loads((Path(source)/'INPUTS.json').read_text())
    data,refs,binding,cache=load_manifest(manifest)
    historical=old['input_binding']
    if (binding['manifest'] != historical['manifest'] or binding['files'] != historical['files']):
        raise ValueError('continuation manifest or original bound inputs changed')
    with refs['context'].open('rb') as f:context=pickle.load(f)
    patterns={cluster:[PatternSpec(p['key'],predicate_from_dict(p['predicate']),
                 None if p['frequency_domain'] is None else ParameterDomain(**p['frequency_domain']),
                 tuple(p['frequency_ops'])) for p in old['Pi'][cluster]] for cluster in old['cluster_order']}
    binding.update(frozen_BO_inputs=file_record(Path(source)/'INPUTS.json'),
                   selector_recomputed=False,patterns_reproposed=False)
    options=dict(old['options'],admission_policy='batch_prefilter',admission_workers=workers,
                 admission_batch_size=batch_size)
    return dict(context=context,si=old['Si'],patterns=patterns,
                relations=tuple(atom_from_dict(a) for a in old['relations']),
                input_binding=binding,lexical_cache_dir=cache,**options)


def _relocated(value):
    if isinstance(value,dict):
        return {k:_relocated(v) for k,v in value.items() if not (k=='path' and 'sha256' in value)}
    if isinstance(value,list):return [_relocated(v) for v in value]
    return value


def _capture_prefix(source, cluster_count, order):
    prefix=[]; offsets=[]; previous_eval=None; ended=False
    for pos in range(cluster_count):
        folder=source/'learning'/('cluster_%02d'%pos)
        if ended:
            offsets.append(0);continue
        before=folder/'admission/before.json'
        after=folder/'admission/after.json'
        if not before.is_file():
            ended=True; offsets.append(0);continue
        start=json.loads(before.read_text())
        if start['rules']!=prefix:raise ValueError('historical cluster prefix differs')
        if after.is_file():
            done=json.loads(after.read_text())
            if done['rules'][:len(prefix)]!=prefix:raise ValueError('historical prefix was removed or reordered')
            prefix=done['rules']; previous_eval=done['evaluation']
            offsets.append(done['all_candidates_processed']);continue
        ordering=json.loads((folder/'admission/ORDER.json').read_text())
        if ordering['order']!=order:raise ValueError('historical partial cluster order differs')
        count=0
        for number in ordering['trial_numbers']:
            p=folder/'admission'/('candidate_%05d.json'%number)
            if not p.is_file():break
            receipt=json.loads(p.read_text())
            trial=json.loads((folder/('trial_%05d.json'%number)).read_text())
            if (receipt['candidate']!=trial['rule'] or receipt['selected_before']!=len(prefix)
                    or receipt['trial_number']!=number or receipt['admission_position']!=count):
                raise ValueError('historical partial receipt does not extend its prefix')
            decision=receipt['decision']
            if decision['accepted']:
                prefix.append(receipt['candidate']); previous_eval=decision['evaluation']
            elif previous_eval is None:
                previous_eval=decision.get('baseline',start['evaluation'])
            if receipt['selected_after']!=len(prefix):raise ValueError('historical selected count differs')
            count+=1
        offsets.append(count);ended=True
    return dict(rules=prefix, consumed_by_cluster=offsets, evaluation=previous_eval,
                scope='historical committed cluster results and contiguous partial receipts; no live score cache imported')


def continue_admission(*, source, output, preserve_prefix=True, resume=False, reuse_BO_standalone=False, **inputs):
    if type(reuse_BO_standalone) is not bool:raise TypeError('standalone reuse flag must be boolean')
    source=Path(source).resolve(strict=True);out=Path(output).resolve()
    if out==source or source in out.parents or out in source.parents:
        raise ValueError('new admission branch must be outside the source run')
    context,si,patterns,relations,binding=(inputs[k] for k in ('context','si','patterns','relations','input_binding'))
    clusters=validate_inputs(context,si,patterns,relations,binding)
    option_names=set(RunOptions.__dataclass_fields__)
    options=RunOptions(**{k:v for k,v in inputs.items() if k in option_names})
    options.validate(len(clusters))
    if options.admission_policy!='batch_prefilter':raise ValueError('continuation requires explicit batch_prefilter')
    old=json.loads((source/'INPUTS.json').read_text())
    current=_run_identity(context,si,patterns,relations,binding,options,clusters,inputs.get('lexical_cache_dir'))
    for key in DATA_KEYS:
        if old.get(key)!=current.get(key):raise ValueError('source/current scientific input differs: '+key)
    for key in ('selection_execution','proof_execution'):
        if _relocated(old[key])!=_relocated(current[key]):
            raise ValueError('source/current execution semantics differ: '+key)
    for key,value in old['options'].items():
        if key not in ('admission_policy','admission_workers','admission_batch_size','workers','initialization_workers') and value!=getattr(options,key):
            raise ValueError('historical BO options differ: '+key)
    # Sources written before the Gamma conflict policy existed were strict.
    if old['options'].get('gamma_conflict_policy','strict')!=options.gamma_conflict_policy:
        raise ValueError('historical Gamma conflict policy differs; set gamma_conflict_policy explicitly')
    if any(not (source/'learning'/('cluster_%02d'%p)/'BO_COMPLETE.json').is_file() for p in range(len(clusters))):
        raise ValueError('all source cluster BO jobs must finish before admission continuation')
    if out.exists() and not resume:raise FileExistsError('new output required or explicit --resume')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable_json(out/'INPUTS.json',current)
        protocol=dict(schema=SCHEMA,source=str(source),source_input_id=digest(old),
            current_input_id=digest(current),preserve_prefix=preserve_prefix,new_BO=0,
            raw_BO_modified=False,approximate=True,admission_order=options.admission_order,
            admission_batch_size=options.admission_batch_size)
        if reuse_BO_standalone:
            from .search import standalone_reuse,committed_prefix
            protocol['standalone_reuse']=dict(schema=standalone_reuse.SCHEMA,
                implementation=file_record(standalone_reuse.__file__),
                continuation_implementation=file_record(__file__),
                prefix_capture_implementation=file_record(committed_prefix.__file__),
                proposed_prefix_decisions_reused=False)
        immutable_json(out/'PROTOCOL.json',protocol)
        # Capture this boundary once. Later source progress cannot alter this branch.
        prefix_path=out/'PRESERVED_PREFIX.json'
        if prefix_path.exists():prefix=json.loads(prefix_path.read_text())
        else:
            prefix=((committed_prefix.capture_committed_prefix(source,len(clusters),options.admission_order,
                        options.admission_batch_size,
                        json.loads((source/'learning/cluster_00/protocol.json').read_text())['evaluator'])
                     if reuse_BO_standalone else _capture_prefix(source,len(clusters),options.admission_order)) if preserve_prefix else
                    dict(rules=[],consumed_by_cluster=[0]*len(clusters),evaluation=None))
            immutable_json(prefix_path,prefix)
        selected=[parse_author_rule(r) for r in prefix['rules']]
        validate_program_rule_forms(selected)
        engine,evaluator,admission=evaluator_pair(context,out,lexical_cache_dir=inputs.get('lexical_cache_dir'),
            gamma_protection=options.gamma_protection,fit_precision_floor=options.precision_floor)
        scorer_class=standalone_reuse.BOStandaloneScorer if reuse_BO_standalone else ParallelScorer
        scorer=scorer_class(context,out,options.admission_workers,inputs.get('lexical_cache_dir'),
            gamma_protection=options.gamma_protection,fit_precision_floor=options.precision_floor)
        try:
            with cached_source_extraction():
                baseline=evaluator.evaluate(selected)
                if prefix['evaluation'] is not None and semantic_result_projection(baseline)!=semantic_result_projection(prefix['evaluation']):
                    raise ValueError('preserved prefix does not reproduce its historical complete facts and score')
                immutable_json(out/'PREFIX_VERIFIED.json',dict(selected=len(selected),evaluation=baseline))
                results=[]
                for pos,cluster in enumerate(clusters):
                    folder=source/'learning'/('cluster_%02d'%pos)
                    complete=json.loads((folder/'BO_COMPLETE.json').read_text())
                    stored=json.loads((folder/'protocol.json').read_text())
                    if complete['protocol_id']!=digest(stored) or complete['complete_trials']!=options.budget_per_cluster:
                        raise ValueError('source BO not complete or changed')
                    records=[json.loads((folder/('trial_%05d.json'%n)).read_text()) for n in range(options.budget_per_cluster)]
                    if complete['all_trial_records_sha256']!=digest(records):raise ValueError('source raw candidates changed')
                    if reuse_BO_standalone:
                        reuse_receipt=scorer.register_cluster(folder,records,evaluator)
                        immutable_json(out/'standalone_reuse'/('cluster_%02d.json'%pos),reuse_receipt)
                    numbers=order_trials(records,options.admission_order)
                    offset=prefix['consumed_by_cluster'][pos]
                    if not 0<=offset<=len(numbers):raise ValueError('invalid preserved count')
                    def progress(record):
                        atomic_json(out/'STATUS.json',dict(record,cluster=cluster,
                            preserved_source_trials=offset,processed_including_preserved=offset+record['processed'],
                            admission_workers=options.admission_workers,seconds=time.monotonic()-started))
                    if offset<len(numbers):
                        selected=admit_batches(order=options.admission_order,
                            directory=out/'learning'/('cluster_%02d'%pos)/'admission',records=records,
                            numbers=numbers[offset:],selected=selected,evaluator=admission,
                            rule_parser=parse_author_rule,score_batch=scorer,
                            batch_size=options.admission_batch_size,on_progress=progress)
                    results.append(dict(cluster=cluster,source=str(folder),preserved_trials=offset,
                                        complete_BO_trials=len(records),selected_after=len(selected)))
                    if reuse_BO_standalone:
                        results[-1].update(source_protocol=file_record(folder/'protocol.json'),
                            source_BO_complete=file_record(folder/'BO_COMPLETE.json'),
                            all_trial_records_sha256=complete['all_trial_records_sha256'])
                final=dict(schema=SCHEMA,rules=[r.to_dict() for r in selected],evaluation=evaluator.evaluate(selected),
                           clusters=results,admission_policy='batch_prefilter',approximate=True,
                           historical_candidates_reused=True,rule_form_audit=validate_program_rule_forms(selected))
                if reuse_BO_standalone:
                    final.update(standalone_reuse_schema=standalone_reuse.SCHEMA,
                        admission_batch_size=options.admission_batch_size,
                        all_candidates_sequentially_evaluated=False,
                        complete_BO_trials=sum(x['complete_BO_trials'] for x in results))
                immutable_json(out/'learning/SELECTED.json',final)
                proof=prove(context,selected,final['evaluation'],out,inputs.get('lexical_cache_dir'),
                            gamma_protection=options.gamma_protection)
                receipt=dict(schema=SCHEMA,status='complete',selected_rules=len(selected),new_BO=0,
                             selection=file_record(out/'learning/SELECTED.json'),proof=file_record(out/'PROOF_REPLAY.json'),
                             final_positive_metrics=final['evaluation']['positive_metrics'],
                             parent_Chase_calls_this_attempt=evaluator.actual_calls,
                             prefilter_Chase_calls_this_attempt=scorer.calls,seconds_this_attempt=time.monotonic()-started)
                if reuse_BO_standalone:
                    archive=dict(schema='loris_same_binding_continuation_source_BO_archive_v1',
                        all_proposed_trials_retained=True,complete_BO_trials=final['complete_BO_trials'],
                        clusters=results,source=str(source),source_inputs=file_record(source/'INPUTS.json'),
                        continuation_protocol=file_record(out/'PROTOCOL.json'),
                        preserved_prefix=file_record(out/'PRESERVED_PREFIX.json'),
                        prefix_verified=file_record(out/'PREFIX_VERIFIED.json'))
                    immutable_json(out/'BO_SOURCE_ARCHIVE.json',archive)
                    receipt.update(archive=file_record(out/'BO_SOURCE_ARCHIVE.json'),
                        complete_BO_trials=final['complete_BO_trials'],standalone_reuse_schema=standalone_reuse.SCHEMA)
                # A completed continuation remains immutable on resume.
                if (out/'COMPLETE.json').is_file():return json.loads((out/'COMPLETE.json').read_text())
                immutable_json(out/'COMPLETE.json',receipt)
                return receipt
        finally:scorer.close();evaluator.close();engine.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--source',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--reuse-bo-standalone',action='store_true')
    p.add_argument('--restart-prefix',action='store_true',help='explicit alternative: re-admit entire historical pool from empty Sigma')
    p.add_argument('--admission-workers',type=int,default=8);p.add_argument('--admission-batch-size',type=int,default=8)
    args=p.parse_args(argv)
    inputs=load_frozen_inputs(args.manifest,args.source,workers=args.admission_workers,
                            batch_size=args.admission_batch_size)
    print(json.dumps(continue_admission(source=args.source,output=args.output,resume=args.resume,
                     preserve_prefix=not args.restart_prefix,reuse_BO_standalone=args.reuse_bo_standalone,**inputs),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
