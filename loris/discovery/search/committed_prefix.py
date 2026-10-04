"""Capture only checksum-committed whole batches from a live BO admission.

Candidate files can precede their batch commit. Such a partial last batch is
not consumed: the new branch will process it with the original batch boundary.
"""
import json
from pathlib import Path
from loris._engine.rlearner import digest
from loris.discovery.manifest import file_record
from .admission import order_trials


def capture_committed_prefix(source, cluster_count, order, batch_size, evaluator_binding):
    source=Path(source);prefix=[];offsets=[];evaluation=None;ended=False;audit=[]
    binding_id=digest(evaluator_binding)
    def key():return digest(dict(binding_id=binding_id,ordered_rules=prefix))
    current_key=key()
    for pos in range(cluster_count):
        folder=source/'learning'/('cluster_%02d'%pos);directory=folder/'admission'
        if ended or not (directory/'before.json').is_file():
            ended=True;offsets.append(0);continue
        before=json.loads((directory/'before.json').read_text())
        if before['rules']!=prefix:raise ValueError('source cluster does not extend committed prefix')
        evaluation=before['evaluation']
        if evaluation['program_key']!=current_key:raise ValueError('source before evaluation has wrong program')
        protocol=json.loads((folder/'protocol.json').read_text())
        complete=json.loads((folder/'BO_COMPLETE.json').read_text())
        records=[json.loads((folder/('trial_%05d.json'%n)).read_text()) for n in range(complete['complete_trials'])]
        if (protocol['evaluator']!=evaluator_binding or complete['protocol_id']!=digest(protocol)
                or complete['all_trial_records_sha256']!=digest(records)):
            raise ValueError('committed prefix requires unchanged same-binding completed BO')
        ordering=json.loads((directory/'ORDER.json').read_text());numbers=order_trials(records,order)
        if (ordering['trial_numbers']!=numbers or ordering['order']!=order
                or ordering['policy']!='batch_prefilter' or ordering['batch_size']!=batch_size
                or ordering['raw_records_sha256']!=digest(records)):
            raise ValueError('source admission order or batch policy differs')
        consumed=0;commits=[]
        for start in range(0,len(numbers),batch_size):
            path=directory/('batch_%05d.json'%(start//batch_size))
            if not path.is_file():break
            saved=json.loads(path.read_text());block=numbers[start:start+batch_size]
            if saved['prefix_key']!=current_key or saved['trial_numbers']!=block:
                raise ValueError('source committed batch prefix or candidate order differs')
            receipts=[]
            for offset,number in enumerate(block):
                receipt=json.loads((directory/('candidate_%05d.json'%number)).read_text())
                if (receipt['candidate']!=records[number]['rule']
                        or receipt['standalone']!=records[number]['standalone']
                        or receipt['trial_number']!=number or receipt['admission_position']!=start+offset
                        or receipt['selected_before']!=len(prefix)
                        or receipt['batch_prefix_key']!=saved['prefix_key']
                        or type(receipt['decision']['accepted']) is not bool):
                    raise ValueError('source committed candidate differs from frozen BO or prefix')
                decision=receipt['decision']
                if decision['accepted']:
                    prefix.append(receipt['candidate']);current_key=key();evaluation=decision['evaluation']
                    if evaluation['program_key']!=current_key:raise ValueError('accepted evaluation has wrong program')
                elif decision.get('baseline',{}).get('program_key')==current_key:
                    evaluation=decision['baseline']
                if receipt['selected_after']!=len(prefix) or receipt['selected_program_key']!=current_key:
                    raise ValueError('source committed candidate has wrong selected program')
                receipts.append(receipt)
            if saved['receipts_sha256']!=digest(receipts) or saved['selected_after']!=len(prefix):
                raise ValueError('source batch checksum or selected count differs')
            consumed+=len(block);commits.append(file_record(path))
        offsets.append(consumed)
        after=directory/'after.json'
        if consumed==len(numbers) and after.is_file():
            final=json.loads(after.read_text())
            if final['rules']!=prefix or final['all_candidates_processed']!=len(numbers):
                raise ValueError('source completed cluster differs from committed batches')
            evaluation=final['evaluation']
        else:ended=True
        audit.append(dict(cluster_position=pos,consumed=consumed,protocol=file_record(folder/'protocol.json'),
            complete_BO=file_record(folder/'BO_COMPLETE.json'),all_trial_records_sha256=digest(records),
            admission_order=file_record(directory/'ORDER.json'),committed_batches=commits))
    return dict(rules=prefix,consumed_by_cluster=offsets,evaluation=evaluation,
                scope='checksum-committed whole batches; any uncommitted last batch is replayed',
                batch_size=batch_size,source_commit_audit=audit)
