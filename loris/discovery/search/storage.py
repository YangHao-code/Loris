"""Typed sampler envelope around unchanged SHA-chained compact persistence."""

import hashlib
import inspect
from pathlib import Path
from loris._engine.compact_storage import (
    STORAGE_MODE,
    FIXED_ATTR,
    ParameterCodec,
    CompactTrial,
    SamplerCheckpointStore as OriginalCheckpointStore,
)
from loris._engine.sparse_bo import JointBinaryTPESampler
from loris._learning.structured_bo import FitWitnessPrior
from loris._engine.primitive_masks import TextPrimitiveCache

MARKER = "_rich_fit_sampler_envelope"


def _binding(kind="RichCombinationTPESampler"):
    module = "random_joint" if kind == "RandomJointSampler" else "sampler"
    return {
        "schema": "paper36_rich_fit_sampler_checkpoint_v1",
        "class": "loris.discovery.search." + module + "." + kind,
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("sampler.py"),
                Path(__file__).with_name("random_joint.py"),
                Path(inspect.getfile(OriginalCheckpointStore)),
                Path(inspect.getfile(JointBinaryTPESampler)),
                Path(inspect.getfile(FitWitnessPrior)),
                Path(inspect.getfile(TextPrimitiveCache)),
            )
        },
    }


class SamplerCheckpointStore(OriginalCheckpointStore):
    def save(self, sampler, completed_trial):
        from .sampler import RichCombinationTPESampler
        from .random_joint import RandomJointSampler

        if type(sampler) not in (RichCombinationTPESampler, RandomJointSampler):
            raise ValueError("checkpoint requires an exact declared discovery sampler class")
        if MARKER in vars(sampler):
            raise ValueError("sampler already carries serialization envelope")
        binding = _binding(type(sampler).__name__)
        envelope = JointBinaryTPESampler.__new__(JointBinaryTPESampler)
        vars(envelope).update(vars(sampler))
        vars(envelope)[MARKER] = binding
        record = super().save(envelope, completed_trial)
        return dict(record, rich_fit_sampler_envelope=binding)

    def restore(self, record):
        from .sampler import RichCombinationTPESampler, SCHEMA
        from .random_joint import RandomJointSampler, SCHEMA as RANDOM_SCHEMA

        kind = record.get("rich_fit_sampler_envelope", {}).get("class", "").rsplit(".", 1)[-1]
        if kind not in ("RichCombinationTPESampler", "RandomJointSampler"):
            raise ValueError("undeclared checkpoint sampler class")
        expected = _binding(kind)
        if record.get("rich_fit_sampler_envelope") != expected:
            raise ValueError("rich sampler source/class binding differs")
        envelope = super().restore(record)
        if vars(envelope).pop(MARKER, None) != expected:
            raise ValueError("rich sampler serialized marker differs")
        cls = RandomJointSampler if kind == "RandomJointSampler" else RichCombinationTPESampler
        sampler = cls.__new__(cls)
        vars(sampler).update(vars(envelope))
        if sampler.protocol.get("schema") != (
            RANDOM_SCHEMA if cls is RandomJointSampler else SCHEMA
        ):
            raise ValueError("serialized proposal policy differs")
        if cls is RichCombinationTPESampler:
            sampler.prior.validate()
            if sampler.prior.binding_id != sampler.protocol["fit_witness_id"]:
                raise ValueError("serialized fit identity differs")
        return sampler
