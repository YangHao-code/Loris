"""AddOnly ablation and historical positive-evidence selection artifacts.

The author-defined main method is implemented in signed_patterns.py. The
canonical CLI defaults to SignedChase; call this implementation explicitly as
AddOnly when reproducing the original positive-union ablation.

All candidates are evaluated and retained. The default is ordered greedy
marginal contribution to the union of previously admitted pattern evidence and
an explicit baseline; independent-baseline scoring is also saved for every
candidate. This additive positive-evidence interpretation, order, label-fitting
rule and thresholds are explicit completions of the paper's missing details.
Coverage is an optional *scientific eligibility policy*, never alleged to be a
lossless prefilter for F1: even coverage-ineligible candidates are fully scored.
Inputs are finite, fully enumerated masks, e.g. from the existing regex/count
store. A frequency instance is one candidate; this API imposes no candidate cap.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zlib
from typing import Mapping

import numpy as np

from loris._engine.rlearner import array_identity, canonical, digest, immutable_json
from loris._engine.rlearner_config import PatternSpec, ParameterDomain
from loris.predicates import FreqPredicate, predicate_from_dict

SCHEMA = 'paper36_aligned_pattern_marginal_f1_v1'


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _pattern_spec(raw, key):
    if not isinstance(raw, Mapping) or set(raw) != {'key', 'predicate', 'frequency_domain', 'frequency_ops'} or raw['key'] != key:
        raise ValueError('spec must be exact PatternSpec.to_dict with matching key')
    domain = ParameterDomain(**raw['frequency_domain']) if raw['frequency_domain'] is not None else None
    spec = PatternSpec(key, predicate_from_dict(raw['predicate']), domain, tuple(raw['frequency_ops']))
    if isinstance(spec.predicate, FreqPredicate):
        values = domain.values or ((domain.low,) if domain.low == domain.high else ())
        if len(values) != 1 or len(spec.frequency_ops) != 1 or values[0] != spec.predicate.eta or spec.frequency_ops[0] != spec.predicate.op:
            raise ValueError('frequency candidates must fix exactly the scored eta/operator; no domain broadening')
    return spec


def _binary(value, shape=None):
    a = np.asarray(value)
    if (shape is not None and a.shape != shape) or not np.isin(a, [0, 1]).all():
        raise ValueError('complete binary array required')
    return np.array(a, dtype=bool, copy=True)


def _rows(values, n):
    rows = list(values)
    if len(rows) != n or any(type(x) not in (str, int) or x == '' for x in rows):
        raise ValueError('complete ordered string/integer row IDs required')
    if len({canonical(x) for x in rows}) != n:
        raise ValueError('row IDs must be unique')
    return rows


def macro_f1(truth, predictions):
    y, p = _binary(truth), _binary(predictions)
    if y.ndim != 2 or y.shape != p.shape or not y.size:
        raise ValueError('matching nonempty row x label matrices required')
    tp = np.count_nonzero(y & p, axis=0)
    den = np.count_nonzero(y, axis=0) + np.count_nonzero(p, axis=0)
    return float(np.divide(2. * tp, den, out=np.zeros(len(den)), where=den != 0).mean())


@dataclass(frozen=True)
class PatternCandidate:
    key: str
    spec: Mapping
    mask: np.ndarray
    label_indices: tuple
    provenance: Mapping


@dataclass(frozen=True)
class PatternSelectionProtocol:
    expected_candidates: int
    min_gain: float = 0.
    strategy: str = 'greedy_marginal'
    min_coverage: float = None
    max_coverage: float = None

    def __post_init__(self):
        if type(self.expected_candidates) is not int or self.expected_candidates < 0:
            raise ValueError('full expected candidate count must be explicit')
        if type(self.min_gain) not in (float, int) or not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError('min_gain must be finite and nonnegative')
        if self.strategy not in ('greedy_marginal', 'independent_marginal'):
            raise ValueError('unknown marginal-F1 strategy')
        for value in (self.min_coverage, self.max_coverage):
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('coverage must be None or finite in [0,1]')
        if self.min_coverage is not None and self.max_coverage is not None and self.min_coverage > self.max_coverage:
            raise ValueError('coverage interval is reversed')


class FitLabelAssigner:
    """Reuse fixed fit sufficient statistics across the entire pattern stream."""
    def __init__(self, truth, *, row_ids, baseline_predictions=None, min_gain=0.):
        y = _binary(truth)
        if y.ndim != 2 or not y.size:
            raise ValueError('nonempty complete fit labels required')
        b = np.zeros_like(y) if baseline_predictions is None else _binary(baseline_predictions, y.shape)
        if type(min_gain) not in (int, float) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError('finite nonnegative fit gain required')
        self.rows = tuple(_rows(row_ids, len(y)))
        self.y = np.frombuffer(y.tobytes(),dtype=bool).reshape(y.shape)
        self.b = np.frombuffer(b.tobytes(),dtype=bool).reshape(b.shape)
        self.positive = np.count_nonzero(y,axis=0)
        self.predicted = np.count_nonzero(b,axis=0)
        self.tp = np.count_nonzero(y & b,axis=0)
        denominator = self.positive+self.predicted
        self.before = np.divide(2.*self.tp,denominator,out=np.zeros(y.shape[1]),where=denominator!=0)
        self.min_gain = min_gain
        self.truth_id, self.baseline_id = array_identity(y), array_identity(b)

    def __call__(self, mask):
        m = _binary(mask,(len(self.y),))
        added = m[:,None] & ~self.b
        tp = self.tp+np.count_nonzero(added & self.y,axis=0)
        denominator = self.positive+self.predicted+np.count_nonzero(added,axis=0)
        after = np.divide(2.*tp,denominator,out=np.zeros(self.y.shape[1]),where=denominator!=0)
        selected = tuple(np.flatnonzero(after-self.before > self.min_gain).tolist())
        # Return independent JSON identities, never mutable aliases to cached
        # metadata; callers may annotate their own candidate provenance.
        return selected, {'role':'fit','label_fit_row_ids':list(self.rows),
            'fit_truth':dict(self.truth_id,shape=list(self.truth_id['shape'])),
            'fit_mask':array_identity(m),
            'fit_baseline':dict(self.baseline_id,shape=list(self.baseline_id['shape'])),
            'fit_min_gain':self.min_gain,
            'label_assignment':'each label with strict positive fit-F1 gain'}


def fit_pattern_labels(mask, truth, *, row_ids, baseline_predictions=None, min_gain=0.):
    """Choose each positive label iff adding the pattern improves its FIT F1.

    The fit rows and their truth identity are returned as provenance. Selection
    rows must be disjoint. Empty labels are retained as a scored no-effect
    candidate. This does not consult selection/control truth to invent heads.
    """
    return FitLabelAssigner(truth,row_ids=row_ids,
        baseline_predictions=baseline_predictions,min_gain=min_gain)(mask)


def select_patterns(candidates, *, truth, baseline_predictions, row_ids, labels,
                    input_binding, protocol, output_dir):
    """Consume every candidate exactly once; save raw/scored/selected SQLite.

    Output is a new run directory. Any interruption keeps partial raw/scored
    rows and no COMPLETE receipt; re-screening may read those stored raw masks
    with ``iter_saved_candidates`` into another directory. No partial result is
    returned as complete. Candidate order is scientific input and is retained.
    """
    if not isinstance(protocol, PatternSelectionProtocol):
        raise ValueError('explicit PatternSelectionProtocol required')
    y = _binary(truth)
    if y.ndim != 2 or not y.size:
        raise ValueError('complete nonempty selection truth required')
    baseline = _binary(baseline_predictions, y.shape)
    rows = _rows(row_ids, len(y))
    labels = list(labels)
    if len(labels) != y.shape[1] or any(type(x) is not str or not x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError('complete unique label axis required')
    if not isinstance(input_binding, Mapping) or input_binding.get('role') != 'control' or type(input_binding.get('expected_rows')) is not int or input_binding.get('expected_rows') != len(y):
        raise ValueError('complete control role/row count required; no test role')
    source = Path(__file__).resolve()
    binding = {'schema': SCHEMA, 'protocol': asdict(protocol), 'input_binding': dict(input_binding),
        'rows': rows, 'labels': labels, 'truth': array_identity(y), 'baseline': array_identity(baseline),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'evidence': 'baseline OR mask times fit-derived positive labels',
        'coverage': 'eligibility policy only; all candidates still scored',
        'order': 'complete caller supplied order; no score sort', 'unknown_is_negative_fact': False}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    immutable_json(directory / 'PROTOCOL.json', binding)
    db = sqlite3.connect(str(directory / 'patterns.sqlite'))
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE fit_axes (id TEXT PRIMARY KEY, row_ids TEXT NOT NULL)')
    db.execute('CREATE TABLE raw (ordinal INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, metadata TEXT NOT NULL, mask BLOB NOT NULL, mask_sha256 TEXT NOT NULL)')
    db.execute('CREATE TABLE scored (ordinal INTEGER PRIMARY KEY REFERENCES raw(ordinal), record TEXT NOT NULL)')
    db.execute('CREATE TABLE selected (ordinal INTEGER PRIMARY KEY REFERENCES raw(ordinal), key TEXT UNIQUE NOT NULL)')
    db.commit()
    selected_keys, seen, count = [], set(), 0
    current = baseline.copy()
    base_score = macro_f1(y, baseline)
    current_score = base_score
    axis = {canonical(x) for x in rows}
    try:
        for ordinal, candidate in enumerate(candidates):
            if ordinal >= protocol.expected_candidates:
                raise ValueError('candidate stream exceeds declared full universe')
            if not isinstance(candidate, PatternCandidate) or type(candidate.key) is not str or not candidate.key or candidate.key in seen:
                raise ValueError('unique complete PatternCandidate key required')
            seen.add(candidate.key)
            if not isinstance(candidate.spec, Mapping) or not isinstance(candidate.provenance, Mapping):
                raise ValueError('pattern spec and fit provenance required')
            _pattern_spec(candidate.spec, candidate.key)
            prov = dict(candidate.provenance)
            fit_rows = prov.get('label_fit_row_ids')
            if prov.get('role') != 'fit' or not isinstance(fit_rows, (list, tuple)) or not fit_rows or 'fit_truth' not in prov:
                raise ValueError('label assignment must bind fit-only truth and rows')
            fit_rows = _rows(fit_rows, len(fit_rows))
            if axis.intersection(canonical(x) for x in fit_rows):
                raise ValueError('fit label-assignment rows overlap control scoring rows')
            inds = list(candidate.label_indices)
            if any(type(i) is not int or not 0 <= i < len(labels) for i in inds) or len(set(inds)) != len(inds):
                raise ValueError('distinct in-range fitted label indices required')
            mask = _binary(candidate.mask, (len(y),))
            bits = np.packbits(mask, bitorder='little').tobytes()
            fit_axis_id = digest(fit_rows)
            stored_prov = dict(prov)
            stored_prov.pop('label_fit_row_ids')
            stored_prov['label_fit_axis_id'] = fit_axis_id
            metadata = {'key': candidate.key, 'spec': dict(candidate.spec), 'label_indices': inds,
                        'provenance': stored_prov, 'mask_rows': len(y)}
            evidence = np.zeros_like(y)
            if inds:
                evidence[:, inds] = mask[:, None]
            independent_score = macro_f1(y, baseline | evidence)
            before = current_score if protocol.strategy == 'greedy_marginal' else base_score
            merged = (current if protocol.strategy == 'greedy_marginal' else baseline) | evidence
            after = macro_f1(y, merged)
            coverage = float(np.count_nonzero(mask) / len(mask))
            eligible = ((protocol.min_coverage is None or coverage >= protocol.min_coverage)
                        and (protocol.max_coverage is None or coverage <= protocol.max_coverage))
            accepted = eligible and after - before > protocol.min_gain
            record = {'ordinal': ordinal, 'key': candidate.key, 'coverage': coverage,
                'coverage_eligible': eligible, 'standalone_baseline_macro_f1': base_score,
                'independent_macro_f1': independent_score, 'independent_gain': independent_score-base_score,
                'marginal_before_macro_f1': before, 'marginal_after_macro_f1': after,
                'marginal_gain': after-before, 'accepted': accepted,
                'reason': 'strict_marginal_gain' if accepted else ('coverage_policy' if not eligible else 'no_strict_marginal_gain')}
            with db:
                db.execute('INSERT OR IGNORE INTO fit_axes VALUES (?,?)', (fit_axis_id, canonical(fit_rows)))
                db.execute('INSERT INTO raw VALUES (?,?,?,?,?)', (ordinal, candidate.key, canonical(metadata), zlib.compress(bits), hashlib.sha256(bits).hexdigest()))
                db.execute('INSERT INTO scored VALUES (?,?)', (ordinal, canonical(record)))
                if accepted:
                    db.execute('INSERT INTO selected VALUES (?,?)', (ordinal, candidate.key))
            if accepted:
                selected_keys.append(candidate.key)
                current |= evidence
                current_score = macro_f1(y, current)
            count += 1
        if count != protocol.expected_candidates:
            raise ValueError('incomplete candidate stream: expected %d, received %d' % (protocol.expected_candidates, count))
    finally:
        db.close()
    if hashlib.sha256(source.read_bytes()).hexdigest() != binding['source_sha256']:
        raise RuntimeError('pattern source changed during evaluation')
    result = {'schema': SCHEMA, 'binding_id': digest(binding), 'raw_candidates': count,
        'scored_candidates': count, 'selected_count': len(selected_keys), 'selected_keys': selected_keys,
        'baseline_macro_f1': base_score, 'final_union_macro_f1': current_score,
        'all_candidates_scored': True, 'coverage_prefilter_saved_work': False,
        'database_sha256': _file_sha(directory/'patterns.sqlite')}
    immutable_json(directory/'SELECTED.json', result)
    immutable_json(directory/'COMPLETE.json', {'schema': SCHEMA, 'result_sha256': digest(result), 'binding_id': digest(binding)})
    return result


def iter_saved_candidates(directory):
    """Recover raw masks, including rejected/zero-gain candidates, in order."""
    db = sqlite3.connect((Path(directory)/'patterns.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        for _, _, raw, packed, expected in db.execute('SELECT * FROM raw ORDER BY ordinal'):
            meta = json.loads(raw)
            bits = zlib.decompress(packed)
            if hashlib.sha256(bits).hexdigest() != expected:
                raise ValueError('saved raw pattern mask differs from its SHA')
            mask = np.unpackbits(np.frombuffer(bits, np.uint8), bitorder='little')[:meta['mask_rows']].astype(bool)
            axis_id = meta['provenance'].pop('label_fit_axis_id')
            axis_record = db.execute('SELECT row_ids FROM fit_axes WHERE id=?', (axis_id,)).fetchone()
            if axis_record is None or digest(json.loads(axis_record[0])) != axis_id:
                raise ValueError('saved fit row axis differs')
            meta['provenance']['label_fit_row_ids'] = json.loads(axis_record[0])
            yield PatternCandidate(meta['key'], meta['spec'], mask, tuple(meta['label_indices']), meta['provenance'])
    finally:
        db.close()


def export_selected_pattern_specs(directory):
    """Verify complete raw/score/selection archive, return ONLY selected specs.

    In particular, a selected frequency instance exports singleton eta/operator
    domains. It never reintroduces the unscreened original frequency template.
    """
    directory = Path(directory)
    protocol = json.loads((directory/'PROTOCOL.json').read_text())
    from .signed_patterns import SCHEMA as SIGNED_SCHEMA, export_selected_signed_pattern_specs
    if protocol.get('schema') == SIGNED_SCHEMA:
        return export_selected_signed_pattern_specs(directory)
    result = json.loads((directory/'SELECTED.json').read_text())
    complete = json.loads((directory/'COMPLETE.json').read_text())
    if (result.get('schema') != SCHEMA or complete.get('schema') != SCHEMA
            or complete.get('result_sha256') != digest(result)
            or complete.get('binding_id') != digest(protocol)
            or result.get('binding_id') != digest(protocol)
            or result.get('database_sha256') != _file_sha(directory/'patterns.sqlite')):
        raise ValueError('pattern selection completion/binding/database checksum differs')
    db = sqlite3.connect((directory/'patterns.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        counts = {table: db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ('raw','scored','selected')}
        if (counts['raw'] != protocol['protocol']['expected_candidates'] or counts['raw'] != result['raw_candidates']
                or counts['scored'] != counts['raw'] or counts['selected'] != result['selected_count']
                or not result.get('all_candidates_scored')):
            raise ValueError('incomplete pattern selection counts')
        output = []
        for ordinal, key, metadata, record in db.execute('SELECT r.ordinal,r.key,r.metadata,s.record FROM raw r JOIN scored s USING(ordinal) JOIN selected t USING(ordinal) ORDER BY r.ordinal'):
            score, raw = json.loads(record), json.loads(metadata)
            if not score.get('accepted') or score.get('ordinal') != ordinal or score.get('key') != key:
                raise ValueError('selected pattern differs from scored admission')
            output.append(_pattern_spec(raw['spec'], key))
        if [p.key for p in output] != result['selected_keys'] or len(output) != counts['selected']:
            raise ValueError('selected pattern axis differs')
        return tuple(output)
    finally:
        db.close()
