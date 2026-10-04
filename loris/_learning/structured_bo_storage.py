"""Explicit typed envelope around the unchanged compact checkpoint journal.

The base object below is only the original store's serialization container. It
is never used to propose a trial. The envelope names and authenticates the real
sampler class before reconstructing it; it does not claim an old BO trajectory.
"""
from pathlib import Path
import hashlib
import inspect
from loris._engine.compact_storage import (STORAGE_MODE, FIXED_ATTR, ParameterCodec,
    CompactTrial, SamplerCheckpointStore as OriginalCheckpointStore)
from loris._engine.sparse_bo import JointBinaryTPESampler

MARKER='_structured_witness_checkpoint_envelope'


def _binding():
    return {'schema':'paper36_structured_witness_checkpoint_envelope_v1',
        'class':'loris._learning.structured_bo.StructuredWitnessTPESampler',
        'sources':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('structured_bo.py'),
             Path(inspect.getfile(OriginalCheckpointStore)),
             Path(inspect.getfile(JointBinaryTPESampler))]}}


class SamplerCheckpointStore(OriginalCheckpointStore):
    def save(self,sampler,completed_trial):
        from .structured_bo import StructuredWitnessTPESampler
        if type(sampler) is not StructuredWitnessTPESampler:
            raise ValueError('structured checkpoint requires its explicitly declared sampler')
        if MARKER in vars(sampler):raise ValueError('sampler already carries a serialization envelope')
        envelope=JointBinaryTPESampler.__new__(JointBinaryTPESampler)
        vars(envelope).update(vars(sampler));vars(envelope)[MARKER]=_binding()
        record=super().save(envelope,completed_trial)
        return dict(record,structured_sampler_envelope=_binding())

    def restore(self,record):
        from .structured_bo import StructuredWitnessTPESampler
        expected=_binding()
        if record.get('structured_sampler_envelope')!=expected:
            raise ValueError('structured sampler checkpoint source/class binding differs')
        envelope=super().restore(record)
        if vars(envelope).pop(MARKER,None)!=expected:
            raise ValueError('structured sampler serialized type marker differs')
        sampler=StructuredWitnessTPESampler.__new__(StructuredWitnessTPESampler)
        vars(sampler).update(vars(envelope))
        if sampler.protocol.get('schema')!='paper36_structured_witness_tpe_v1':
            raise ValueError('serialized proposal policy differs')
        sampler.prior.validate()
        if sampler.prior.binding_id!=sampler.protocol['fit_witness_id']:
            raise ValueError('serialized fit witness identity differs')
        return sampler
