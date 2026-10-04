"""Lossless storage for the explicit full-indicator JointBinary BO protocol.

Zero indicators are implicit only on disk. The complete ordered manifest,
full sampled vectors, conditional parameters and original RNG are retained.
No proposal distribution, feature support, budget or score is changed here.
Trusted local pickle snapshots are SHA checked before restoration.
"""
from __future__ import annotations

import hashlib
import json
import operator
import os
from pathlib import Path
import pickle

import numpy as np

from loris._engine.discovery_io import sync_file


PARAM_SCHEMA = 'paper36_lossless_zero_default_parameters_v1'
STORAGE_MODE = 'compact_full_indicators_v1'
PARAM_ATTR = 'paper36_compact_parameters_v1'
FIXED_ATTR = 'paper36_compact_fixed_parameters_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _immutable_bytes(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('immutable compact artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('wb') as stream:
        stream.write(raw)
        sync_file(stream)
    os.replace(str(tmp), str(path))


class ParameterCodec:
    """One manifest declares every bit; sorted ON indices have no length cap."""
    def __init__(self, include_keys, space_id):
        self.include_keys = tuple(include_keys)
        if not self.include_keys or len(set(self.include_keys)) != len(self.include_keys):
            raise ValueError('complete unique include axis required')
        self.key_set = frozenset(self.include_keys)
        self.dimension = len(self.include_keys)
        self.space_id = space_id
        self.axis_sha256 = _sha(_json(list(self.include_keys)).encode())

    @classmethod
    def from_space(cls, space):
        return cls((f.include_key for f in space.features), space.space_id)

    def encode(self, params):
        fast = self._encode_fast(params)
        if fast is not None:
            return fast
        on = []
        for index, key in enumerate(self.include_keys):
            value = params.get(key)
            if type(value) is not int or value not in (0, 1):
                raise ValueError('every include bit must be an explicit integer zero or one')
            if value:
                on.append(index)
        active = {k: v for k, v in params.items() if k not in self.key_set}
        result = {'schema': PARAM_SCHEMA, 'space_id': self.space_id,
                  'include_axis_sha256': self.axis_sha256, 'dimension': self.dimension,
                  'default': 0, 'on_indices': on, 'active_parameters': active}
        _json(result)  # Reject nonfinite/non-JSON parameters before persistence.
        return result

    def _encode_fast(self, params):
        """Same result as the loop in encode(), or None to run that loop."""
        if type(params) is not dict:
            return None
        getter = self.__dict__.get('_getter')
        if getter is None:
            getter = operator.itemgetter(*self.include_keys)
            self._getter = getter
        try:
            values = getter(params)
        except KeyError:
            return None
        if self.dimension == 1:
            values = (values,)
        if set(map(type, values)) != {int} or not set(values) <= {0, 1}:
            return None
        on = np.flatnonzero(np.fromiter(values, np.int8, self.dimension)).tolist()
        extra = params.keys() - self.key_set
        active = {k: v for k, v in params.items() if k in extra} if extra else {}
        result = {'schema': PARAM_SCHEMA, 'space_id': self.space_id,
                  'include_axis_sha256': self.axis_sha256, 'dimension': self.dimension,
                  'default': 0, 'on_indices': on, 'active_parameters': active}
        _json(result)  # Reject nonfinite/non-JSON parameters before persistence.
        return result

    def validate(self, value):
        if (set(value) != {'schema', 'space_id', 'include_axis_sha256', 'dimension',
                           'default', 'on_indices', 'active_parameters'}
                or value['schema'] != PARAM_SCHEMA or value['space_id'] != self.space_id
                or value['include_axis_sha256'] != self.axis_sha256
                or type(value['dimension']) is not int or value['dimension'] != self.dimension
                or type(value['default']) is not int or value['default'] != 0):
            raise ValueError('compact parameters differ from complete manifest binding')
        indices = value['on_indices']
        active = value['active_parameters']
        if (not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < self.dimension for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or not isinstance(active, dict) or self.key_set.intersection(active)):
            raise ValueError('invalid compact ON indices or conditional parameters')
        _json(value)
        return value

    def decode(self, value):
        self.validate(value)
        params = dict.fromkeys(self.include_keys, 0)
        for i in value['on_indices']:
            params[self.include_keys[i]] = 1
        params.update(value['active_parameters'])
        return params

    def packed_mask(self, value):
        self.validate(value)
        mask = np.zeros(self.dimension, np.uint8)
        mask[value['on_indices']] = 1
        return np.packbits(mask)

    def frozen_parameters(self, trial):
        value = trial.user_attrs.get(PARAM_ATTR)
        if value is None:
            raise ValueError('Optuna trial lacks sealed compact parameters')
        self.validate(value)
        if trial.params != value['active_parameters']:
            raise ValueError('Optuna conditional parameters differ from compact receipt')
        return value


class CompactTrial:
    """Skip per-bit Optuna writes, retaining exactly the same joint proposal.

    Conditional/head suggestions still use the original real Trial and TPE.
    Enqueued fixed vectors are served without advancing either sampler RNG.
    """
    def __init__(self, trial, codec):
        self.trial, self.codec = trial, codec
        self.params = {}
        self.fixed = trial.user_attrs.get(FIXED_ATTR)
        self.fixed_params = codec.decode(self.fixed) if self.fixed is not None else None

    def suggest_categorical(self, name, choices):
        if name in self.codec.key_set:
            if list(choices) != [0, 1]:
                raise ValueError('compact include distribution must be [0, 1]')
            if name in self.params:
                return self.params[name]
            value = (self.fixed_params[name] if self.fixed_params is not None
                     else self.trial.relative_params[name])
            if type(value) is not int or value not in (0, 1):
                raise ValueError('joint sampler returned an invalid include bit')
        else:
            value = self.trial.suggest_categorical(name, choices)
        self.params[name] = value
        return value

    def include_bits(self, keys):
        """Bulk read of include bits with the exact per-key source and checks.

        Returns None (caller falls back to per-key suggestions, reproducing any
        original error at the original key) unless every bit is a fresh,
        explicit int 0/1. Reading has no side effect; adopt_include_bits
        records values exactly as suggest_categorical would.
        """
        if not keys or not self.params.keys().isdisjoint(keys):
            return None
        source = self.fixed_params if self.fixed_params is not None else self.trial.relative_params
        try:
            values = operator.itemgetter(*keys)(source)
        except KeyError:
            return None
        if len(keys) == 1:
            values = (values,)
        if set(map(type, values)) != {int} or not set(values) <= {0, 1}:
            return None
        return values

    def adopt_include_bits(self, keys, values):
        if any(key not in self.codec.key_set for key in keys[:1]) or len(keys) != len(values):
            raise ValueError('bulk include adoption outside the complete include axis')
        self.params.update(zip(keys, values))

    def suggest_float(self, name, *args, **kwargs):
        value = self.trial.suggest_float(name, *args, **kwargs)
        self.params[name] = value
        return value

    def suggest_int(self, name, *args, **kwargs):
        value = self.trial.suggest_int(name, *args, **kwargs)
        self.params[name] = value
        return value

    def seal(self):
        value = self.codec.encode(self.params)
        if value['active_parameters'] != self.trial.params:
            raise ValueError('suggested conditional parameters differ from real Trial')
        if self.fixed is not None and value != self.fixed:
            raise ValueError('enqueued compact parameters were not consumed exactly')
        self.trial.set_user_attr(PARAM_ATTR, value)
        return value


def _small_plain_equal(a, b, budget=None):
    """Equality only for small trees of JSON-like builtins; otherwise False."""
    budget = [4096] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or type(a) is not type(b):
        return False
    if a is None or type(a) in (str, int, float, bool):
        return a == b
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_small_plain_equal(x, y, budget) for x, y in zip(a, b))
    if type(a) is dict:
        return (list(a) == list(b)
                and all(_small_plain_equal(a[k], b[k], budget) for k in a))
    return False


class SamplerCheckpointStore:
    """Static state once, RNG state per checkpoint, sparse observations once.

    Each snapshot references a SHA chained observation prefix. Restoring a
    pre-tell snapshot ignores subsequent committed journal entries, so the
    callback can be replayed exactly once after a split transaction crash.
    History is reconstructed losslessly into the ORIGINAL dense sampler API.
    """
    _dynamic = frozenset(('rng', 'delegate', 'observations', 'last_diagnostic'))

    # Exact static-snapshot reuse (2026-09-30): the static partition is
    # pickled once and re-pickled only when one of its values is replaced.
    # Values are compared by identity (the cache holds strong references, so
    # an identity cannot be recycled) or, for small plain JSON-like values,
    # by equality. Every STATIC_VERIFY_EVERY saves the complete static state
    # is re-pickled and must reproduce the cached SHA; any in-place mutation
    # therefore fails loudly instead of silently referencing an old snapshot.
    STATIC_VERIFY_EVERY = 256

    def __init__(self, directory, codec):
        self.directory, self.codec = Path(directory), codec
        self._root = _sha(b'paper36 compact sampler observations v1')
        self._count = 0
        self._known = {}
        self._static_cache = None
        self._static_saves = 0

    def _write_blob(self, value):
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        sha = _sha(raw)
        _immutable_bytes(self.directory / 'states' / (sha + '.pickle'), raw)
        return sha

    def _read_blob(self, sha):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid checkpoint SHA reference')
        raw = (self.directory / 'states' / (sha + '.pickle')).read_bytes()
        if _sha(raw) != sha:
            raise ValueError('sampler checkpoint checksum differs')
        return pickle.loads(raw)

    def _observation_path(self, number):
        return self.directory / 'observations' / ('%05d.json' % number)

    def save(self, sampler, completed_trial):
        from loris._engine.sparse_bo import JointBinaryTPESampler
        if type(sampler) is not JointBinaryTPESampler:
            raise ValueError('compact checkpoints require the exact declared JointBinaryTPESampler')
        if sampler.include_keys != self.codec.include_keys:
            raise ValueError('sampler checkpoint include axis differs')
        count = completed_trial + 1
        if set(sampler.observations) != set(range(count)):
            raise ValueError('sampler observations must be the complete consecutive trial prefix')
        if count < self._count:
            raise ValueError('restore the earlier checkpoint before saving an earlier prefix')
        # Existing observations cannot mutate in the sampler's checked _record.
        # New events are immutable; recovery also checks every event's SHA chain.
        for number in range(self._count, count):
            score, packed = sampler.observations[number]
            mask = np.unpackbits(packed)[:self.codec.dimension]
            event = {'number': number, 'score': score, 'dimension': self.codec.dimension,
                     'include_axis_sha256': self.codec.axis_sha256,
                     'on_indices': np.flatnonzero(mask).tolist(), 'previous_sha256': self._root}
            raw = (_json(event) + '\n').encode()
            _immutable_bytes(self._observation_path(number), raw)
            self._root = _sha(raw)
            self._known[number] = (score, packed)
            self._count += 1
        static = {k: v for k, v in vars(sampler).items() if k not in self._dynamic}
        dynamic = {k: v for k, v in vars(sampler).items() if k in self._dynamic and k != 'observations'}
        return {'schema': 'paper36_lossless_sampler_checkpoint_v1',
                'completed_trial': completed_trial, 'static_sha256': self._static_sha(static),
                'state_sha256': self._write_blob(dynamic), 'observation_count': count,
                'observation_prefix_sha256': self._root}

    def _static_sha(self, static):
        self._static_saves += 1
        cached = self._static_cache
        reusable = (cached is not None and list(static) == list(cached[0])
                    and all(static[k] is cached[0][k] or _small_plain_equal(static[k], cached[0][k])
                            for k in static)
                    and (self.directory / 'states' / (cached[1] + '.pickle')).is_file())
        if reusable and self._static_saves % self.STATIC_VERIFY_EVERY:
            return cached[1]
        sha = self._write_blob(static)
        if reusable and sha != cached[1]:
            raise RuntimeError('static sampler state was mutated in place; checkpoint identity differs')
        self._static_cache = (dict(static), sha)
        return sha

    def restore(self, record):
        from loris._engine.sparse_bo import JointBinaryTPESampler
        if (record.get('schema') != 'paper36_lossless_sampler_checkpoint_v1'
                or type(record.get('completed_trial')) is not int
                or record.get('observation_count') != record['completed_trial'] + 1
                or record['observation_count'] < 0):
            raise ValueError('invalid compact sampler checkpoint')
        sampler = JointBinaryTPESampler.__new__(JointBinaryTPESampler)
        static = self._read_blob(record['static_sha256'])
        dynamic = self._read_blob(record['state_sha256'])
        if self._dynamic.intersection(static) or set(dynamic) != self._dynamic - {'observations'}:
            raise ValueError('invalid static/dynamic sampler state partition')
        vars(sampler).update(static)
        vars(sampler).update(dynamic)
        if sampler.include_keys != self.codec.include_keys:
            raise ValueError('sampler checkpoint include axis differs')
        root = _sha(b'paper36 compact sampler observations v1')
        observations = {}
        for number in range(record['observation_count']):
            raw = self._observation_path(number).read_bytes()
            event = json.loads(raw)
            indices = event['on_indices']
            if (event['number'] != number or event['previous_sha256'] != root
                    or event['dimension'] != self.codec.dimension
                    or event['include_axis_sha256'] != self.codec.axis_sha256
                    or not np.isfinite(event['score']) or not isinstance(indices, list)
                    or any(type(i) is not int or not 0 <= i < self.codec.dimension for i in indices)
                    or any(a >= b for a, b in zip(indices, indices[1:]))):
                raise ValueError('sampler observation journal differs')
            mask = np.zeros(self.codec.dimension, np.uint8)
            mask[indices] = 1
            observations[number] = (float(event['score']),
                                    np.frombuffer(np.packbits(mask).tobytes(), np.uint8))
            root = _sha(raw)
        if root != record['observation_prefix_sha256']:
            raise ValueError('sampler observation prefix checksum differs')
        sampler.observations = observations
        self._root, self._count, self._known = root, len(observations), dict(observations)
        return sampler
