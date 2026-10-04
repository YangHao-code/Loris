"""Complete finite pattern enumeration with exact, reusable screening.

The anchor extractor defines an explicit finite experimental vocabulary. For
EVERY anchor this module screens match and every distinct frequency partition;
for EVERY unordered pair it screens cooccur and both before directions. There
is no pair cap or early entropy/maximum-coverage pruning of anchors. A pair can
be omitted from physical enumeration only when its exact support upper bound
proves failure of the declared minimum coverage. The full raw universe remains
reconstructible from the anchor catalog, enumerator and saved certificates.

PDF36 does not specify the numerical coverage/entropy thresholds or a unique
multilabel entropy normalization. Both are explicit here, not alleged defaults.
All counts/positions cover the complete supplied corpus; only the declared fit
cluster's gold labels are used for pattern screening. No old rule is read.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import zlib

import numpy as np

from loris.predicates import MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate
from loris.predicates._core import _Pattern
from loris._engine.rlearner import atomic_json, immutable_json, digest, array_identity
from loris._engine.rlearner_config import PatternSpec, ParameterDomain, FREQUENCY_OPS
from loris._engine.regex_superset_index import RegexSupersetIndex
from loris._engine.shared_literal_stats import SharedLiteralStats, configuration as shared_configuration


SCHEMA = 'paper36_complete_finite_pattern_screening_v1'


def pack_record(record):
    return zlib.compress(json.dumps(record, separators=(',', ':'), ensure_ascii=False).encode('utf8'), 6)


def unpack_record(record):
    return json.loads(zlib.decompress(record).decode('utf8'))


def minimum_support(n, coverage):
    # ceil(n*c) can be one too high due to floating multiplication even when
    # support/n == c exactly. Match the actual coverage predicate, not rounded
    # integer arithmetic used only as an optimization.
    result = int(math.ceil(n * coverage))
    while result > 0 and (result - 1) / n >= coverage:
        result -= 1
    while result / n < coverage:
        result += 1
    return result


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def literal_regex(groups):
    return r'\s+'.join(r'\b(?:' + '|'.join(re.escape(word) for word in
        sorted(set(group), key=lambda word: (-len(word), word))) + r')\b' for group in groups)


class LiteralDocumentIndex:
    """An exact SUPERSET filter; final regex finditer always decides matches.

    ASCII substrings intentionally overinclude words adjoining Unicode letters.
    The four non-ASCII characters matched by ASCII letters under Python's
    IGNORECASE are always included. Thus Unicode word boundaries and case
    behavior cannot silently remove real matches. Unsupported structure scans
    all rows. This index never decides an actual count or first position.
    """
    def __init__(self, texts):
        self.texts = tuple(texts)
        self.postings = defaultdict(set)
        self.special = set()
        for row, text in enumerate(self.texts):
            for word in set(re.findall(r'[A-Za-z0-9_]+', text)):
                self.postings[word.lower()].add(row)
            if any(ch in text for ch in '\u0130\u0131\u017f\u212a'):
                self.special.add(row)

    def candidate_rows(self, raw, flags, groups):
        if (not groups or flags not in (0, re.IGNORECASE, re.UNICODE, re.IGNORECASE | re.UNICODE)
                or any(not group or any(not isinstance(word, str) or not re.fullmatch(r'[A-Za-z0-9_]+', word)
                    for word in group) for group in groups)
                or raw != literal_regex(groups)):
            return range(len(self.texts))
        candidates = None
        for group in groups:
            union = set()
            for word in group:
                union.update(self.postings.get(word.lower(), ()))
            candidates = union if candidates is None else candidates.intersection(union)
        if flags & re.IGNORECASE:
            candidates.update(self.special)
        return sorted(candidates)


class CompleteRegexStore:
    """SHA-checked compressed count/first arrays; no eviction of raw results."""
    def __init__(self, docs, directory, attribute_presence=None, *, regex_index='ast',
                 shared_literal_policy=None, shared_literal_bytes=None):
        if regex_index not in ('ast', 'literal'):
            raise ValueError('explicit supported exact superset index required')
        self.index_class = RegexSupersetIndex if regex_index == 'ast' else LiteralDocumentIndex
        self.values = {attr: tuple(doc.get_attr(attr) for doc in docs) for attr in ('cnt', 'ttl', 'mtd')}
        self.n = len(docs)
        if any(not isinstance(text, str) for values in self.values.values() for text in values):
            raise ValueError('explicit complete text strings required')
        self.presence = {}
        for attr, values in self.values.items():
            mask = np.asarray((attribute_presence or {}).get(attr, [v != '' for v in values]))
            if mask.shape != (self.n,) or not np.isin(mask, [0, 1]).all():
                raise ValueError('presence must cover all source rows')
            self.presence[attr] = mask.astype(bool)
        shared_config = shared_configuration(shared_literal_policy, shared_literal_bytes)
        self.binding = {'schema': SCHEMA + '_regex', 'rows': self.n,
            'texts': {k: digest(v) for k, v in self.values.items()},
            'presence': {k: array_identity(v) for k, v in self.presence.items()},
            'implementation_sha256': file_sha(__file__), 'regex_index': regex_index,
            'regex_index_source_sha256': file_sha(Path(__file__).with_name('regex_superset_index.py')) if regex_index == 'ast' else None,
            'shared_literal': shared_config,
            'shared_literal_source_sha256': file_sha(Path(__file__).with_name('shared_literal_stats.py'))}
        self.identity = digest(self.binding)
        directory = Path(directory) / self.identity
        immutable_json(directory / 'binding.json', self.binding)
        self.db = sqlite3.connect(str(directory / 'regex.sqlite'), timeout=120)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS regexes(key TEXT PRIMARY KEY, definition TEXT, payload BLOB, sha256 TEXT)')
        self.indices = {}
        self.stats = Counter()
        self.shared_literals = SharedLiteralStats(self.values, config=shared_config, stats=self.stats)

    def counts_first(self, attr, anchor):
        if attr not in self.values:
            raise ValueError('only declared text attributes are supported')
        raw, flags = anchor['raw'], anchor['flags']
        if not isinstance(raw, str) or type(flags) is not int:
            raise ValueError('explicit raw regex and integer flags required')
        definition = [attr, raw, flags]
        key = digest(definition)
        saved = self.db.execute('SELECT definition,payload,sha256 FROM regexes WHERE key=?', (key,)).fetchone()
        if saved:
            encoded, payload, sha = saved
            if json.loads(encoded) != definition or hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError('saved complete regex statistics changed')
            result = np.load(io.BytesIO(zlib.decompress(payload)), allow_pickle=False)
            self.stats['cache_hits'] += 1
        else:
            if attr not in self.indices:
                self.indices[attr] = self.index_class(self.values[attr])
            candidates = self.indices[attr].candidate_rows(raw, flags, anchor.get('literal_alternatives_by_token'))
            shared_args = {'candidate_rows': candidates}
            if 'literal_alternatives_by_token' in anchor:
                shared_args['metadata'] = anchor['literal_alternatives_by_token']
            result = self.shared_literals.maybe_counts(attr, raw, flags, **shared_args)
            memo = {}
            if result is None:
                result = np.zeros((2, self.n), np.int64)
                result[1] = -1
                compiled = re.compile(raw, flags)
                for row in candidates:
                    text = self.values[attr][row]
                    found = memo.get(text)
                    if found is None:
                        first, count = -1, 0
                        for match in compiled.finditer(text):
                            if count == 0:
                                first = match.start()
                            count += 1
                        found = (count, first)
                        memo[text] = found
                    result[:, row] = found
            stream = io.BytesIO()
            np.save(stream, result, allow_pickle=False)
            payload = zlib.compress(stream.getvalue(), 6)
            self.db.execute('INSERT INTO regexes VALUES(?,?,?,?)',
                (key, json.dumps(definition), payload, hashlib.sha256(payload).hexdigest()))
            self.stats['computed_regexes'] += 1
            self.stats['candidate_rows'] += len(candidates)
            self.stats['distinct_text_scans'] += len(memo)
            self.stats['proved_nonmatching_rows'] += self.n - len(candidates)
        if result.shape != (2, self.n) or result.dtype != np.int64:
            raise ValueError('incomplete saved count/position array')
        result.flags.writeable = False
        return result

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()


@dataclass(frozen=True)
class ScreeningProtocol:
    min_coverage: float
    max_coverage: float
    max_entropy: float

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (self.min_coverage, self.max_coverage, self.max_entropy)):
            raise ValueError('finite explicit screening thresholds required')
        if not 0 < self.min_coverage <= self.max_coverage <= 1 or self.max_entropy <= 0:
            raise ValueError('invalid screening threshold range')

    def manifest(self):
        return dict(asdict(self),
            coverage='matching rows / ALL rows in this fit cluster, including missing fields',
            entropy='natural-log entropy of normalized total positive label-occurrence histogram; empty histogram -> 0',
            entropy_comparison='strictly less than max_entropy',
            frequency='all six operators, every integer threshold 0..max_observed_fit_cluster_count+1',
            frequency_admission='retain template if ANY allowed parameter passes, not only a prototype',
            pairs='all unordered anchor pairs, cooccur and both before directions',
            sound_pair_pruning='only min(exact individual match supports) < minimum required support',
            inherited_rule_or_pattern_inputs=False, hidden_pair_or_anchor_caps=False)


def mask_metrics(mask, truth, protocol):
    mask = np.asarray(mask, bool)
    support = int(mask.sum())
    coverage = support / len(mask)
    histogram = np.asarray(truth[mask].sum(axis=0), np.int64)
    total = int(histogram.sum())
    positive = histogram[histogram > 0].astype(np.float64)
    entropy = float(-np.sum((positive / total) * np.log(positive / total))) if total else 0.
    reason = ('below_min_coverage' if coverage < protocol.min_coverage else
              'above_max_coverage' if coverage > protocol.max_coverage else
              'entropy' if entropy >= protocol.max_entropy else 'accepted')
    return {'support': support, 'coverage': coverage, 'entropy': entropy,
            'label_histogram': histogram.tolist(), 'accepted': reason == 'accepted', 'reason': reason}


_OPERATIONS = {'==': np.equal, '!=': np.not_equal, '<': np.less,
               '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal}


def screen_cluster_field(directory, *, store, anchors, cluster_rows, fit_truth,
                         label_names, attr, cluster_id, protocol, provenance, on_progress=None):
    """Durably screen the COMPLETE declared finite universe for one field."""
    if not isinstance(protocol, ScreeningProtocol):
        raise ValueError('explicit ScreeningProtocol required')
    rows = np.asarray(cluster_rows)
    truth = np.asarray(fit_truth)
    if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer) or len(set(rows.tolist())) != len(rows):
        raise ValueError('unique nonempty integer cluster row axis required')
    if np.any((rows < 0) | (rows >= store.n)) or truth.shape != (len(rows), len(label_names)) or not np.isin(truth, [0, 1]).all():
        raise ValueError('complete fit-cluster truth and source rows required')
    if not anchors or len({a['key'] for a in anchors}) != len(anchors):
        raise ValueError('nonempty unique anchor catalog required')
    directory = Path(directory)
    binding = {'schema': SCHEMA, 'cluster_id': cluster_id, 'attr': attr,
        'store': store.binding, 'row_axis': rows.tolist(), 'fit_truth': array_identity(truth),
        'label_names': list(label_names), 'anchors_sha256': digest(anchors),
        'screening': protocol.manifest(), 'provenance': provenance}
    immutable_json(directory / 'protocol.json', binding)
    immutable_json(directory / 'anchors.json', {'anchors': anchors})
    identity = digest(binding)
    done = directory / 'COMPLETE.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['protocol_id'] != identity or any(file_sha(directory / name) != sha for name, sha in receipt['files'].items()):
            raise ValueError('completed pattern screening changed')
        return receipt
    started = time.perf_counter()
    db = sqlite3.connect(str(directory / 'screening.sqlite'), timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS singles(i INTEGER PRIMARY KEY, record BLOB)')
    db.execute('CREATE TABLE IF NOT EXISTS pairs(i INTEGER,j INTEGER,record BLOB,PRIMARY KEY(i,j))')
    db.execute('CREATE TABLE IF NOT EXISTS blocks(i INTEGER PRIMARY KEY, pairs INTEGER)')
    counts, firsts, supports = [], [], []
    selected = []

    def add_spec(key, predicate, domain=None):
        selected.append(PatternSpec(key, predicate, domain).to_dict())

    def pattern(i):
        return _Pattern(anchors[i]['raw'], anchors[i]['flags'])

    try:
        for i, anchor in enumerate(anchors):
            values = store.counts_first(attr, anchor)
            count, first = values[0, rows], values[1, rows]
            present = store.presence[attr][rows]
            mask = (count > 0) & present
            counts.append(count)
            firsts.append(first)
            supports.append(int(mask.sum()))
            previous = db.execute('SELECT record FROM singles WHERE i=?', (i,)).fetchone()
            if previous:
                record = unpack_record(previous[0])
            else:
                match = mask_metrics(mask, truth, protocol)
                frequency, memo = [], {}
                maximum = int(count.max()) + 1
                for op in FREQUENCY_OPS:
                    for eta in range(maximum + 1):
                        freq_mask = _OPERATIONS[op](count, eta) & present
                        signature = np.packbits(freq_mask).tobytes()
                        if signature not in memo:
                            memo[signature] = mask_metrics(freq_mask, truth, protocol)
                        frequency.append({'op': op, 'eta': eta, **memo[signature]})
                record = {'i': i, 'key': anchor['key'], 'match': match,
                    'frequency': frequency, 'frequency_domain_high': maximum}
                db.execute('INSERT INTO singles VALUES(?,?)', (i, pack_record(record)))
            if record['match']['accepted']:
                add_spec('%s_match_%s' % (attr, anchor['key']), MatchPredicate(attr, pattern(i)))
            accepted = next((r for r in record['frequency'] if r['accepted']), None)
            if accepted is not None:
                add_spec('%s_freq_%s' % (attr, anchor['key']),
                    FreqPredicate(attr, pattern(i), accepted['op'], accepted['eta']),
                    ParameterDomain(low=0, high=record['frequency_domain_high'], integer=True))
            if (i + 1) % 200 == 0 or i + 1 == len(anchors):
                store.commit()
                db.commit()
                status = {'phase': 'singles', 'cluster_id': cluster_id, 'attr': attr,
                    'completed': i + 1, 'total': len(anchors), 'selected': len(selected),
                    'elapsed_seconds': time.perf_counter() - started, 'regex_store': dict(store.stats)}
                atomic_json(directory / 'progress.json', status)
                if on_progress:
                    on_progress(status)
        supports = np.asarray(supports, np.int64)
        minimum = minimum_support(len(rows), protocol.min_coverage)
        eligible = np.flatnonzero(supports >= minimum).tolist()
        full_pairs = len(anchors) * (len(anchors) - 1) // 2
        evaluated_pairs = len(eligible) * (len(eligible) - 1) // 2
        universe = {'schema': SCHEMA + '_complete_raw_universe', 'anchors': len(anchors),
            'match_templates': len(anchors), 'frequency_templates': len(anchors),
            'unordered_pairs': full_pairs, 'pair_templates': 3 * full_pairs,
            'enumeration_order': 'anchor catalog order; match/freq per anchor; i<j cooccur,before(i,j),before(j,i)',
            'minimum_required_support': minimum, 'exact_individual_match_support': supports.tolist(),
            'physically_evaluated_pair_indices': eligible, 'physically_evaluated_pairs': evaluated_pairs,
            'certified_rejected_pairs': full_pairs - evaluated_pairs,
            'rejection_certificate': 'For every omitted i<j, one exact support is below minimum_required_support. cooccur and either before mask are subsets of BOTH match masks.',
            'no_pruning_on_single_entropy_or_max_coverage': True,
            'reconstruct_all_raw_templates': 'anchors.json + this enumeration; frequency parameter domains/decisions in screening.sqlite singles',
            'raw_templates_including_certified_rejections': 2 * len(anchors) + 3 * full_pairs}
        immutable_json(directory / 'raw_universe.json', universe)
        matched = [(counts[i] > 0) & store.presence[attr][rows] for i in range(len(anchors))]
        for offset, i in enumerate(eligible):
            committed = db.execute('SELECT pairs FROM blocks WHERE i=?', (i,)).fetchone()
            if committed:
                if committed[0] != len(eligible) - offset - 1:
                    raise ValueError('committed pair block is incomplete')
                block = [(int(j), unpack_record(record)) for j, record in db.execute(
                    'SELECT j,record FROM pairs WHERE i=? ORDER BY j', (i,))]
                if [j for j, _ in block] != eligible[offset + 1:]:
                    raise ValueError('committed pair rows differ')
            else:
                block = []
                for j in eligible[offset + 1:]:
                    both = matched[i] & matched[j]
                    co = mask_metrics(both, truth, protocol)
                    # A failing pair minimum proves BOTH orderings fail too.
                    # Otherwise evaluate each exactly, even if co's entropy or
                    # maximum coverage fails; its ordering can still pass.
                    if co['support'] < minimum:
                        before = [{'accepted': False, 'reason': 'certified_by_cooccur_support',
                            'support_upper_bound': co['support']} for _ in range(2)]
                    else:
                        before = [mask_metrics(both & (firsts[a] < firsts[b]), truth, protocol)
                                  for a, b in ((i, j), (j, i))]
                    record = {'cooccur': co, 'before_ab': before[0], 'before_ba': before[1]}
                    block.append((j, record))
                    db.execute('INSERT INTO pairs VALUES(?,?,?)',
                        (i, j, pack_record(record)))
                db.execute('INSERT INTO blocks VALUES(?,?)', (i, len(block)))
                db.commit()
            for j, record in block:
                key = '%s_%s_%s' % (attr, anchors[i]['key'], anchors[j]['key'])
                if record['cooccur']['accepted']:
                    add_spec('co_' + key, CooccurPredicate(attr, pattern(i), pattern(j)))
                if record['before_ab']['accepted']:
                    add_spec('ab_' + key, BeforePredicate(attr, pattern(i), pattern(j)))
                if record['before_ba']['accepted']:
                    add_spec('ba_' + key, BeforePredicate(attr, pattern(j), pattern(i)))
            status = {'phase': 'pairs', 'cluster_id': cluster_id, 'attr': attr,
                'completed_anchor_blocks': offset + 1, 'total_anchor_blocks': len(eligible),
                'selected': len(selected), 'elapsed_seconds': time.perf_counter() - started}
            atomic_json(directory / 'progress.json', status)
            if on_progress and (offset % 25 == 0 or offset + 1 == len(eligible)):
                on_progress(status)
        store.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.close()
        db = None
        spec_path = directory / 'pattern_specs.json'
        immutable_json(spec_path, {'schema': 'paper36_pattern_specs_v1', 'label_names': list(label_names),
            'patterns': selected, 'provenance': {'protocol_id': identity,
                'current_complete_fit_cluster': True, 'fresh_anchor_catalog_sha256': digest(anchors),
                'full_raw_universe': universe, 'screening': protocol.manifest()}})
        result = {'status': 'complete', 'protocol_id': identity, 'cluster_id': cluster_id, 'attr': attr,
            'fit_rows': len(rows), 'full_source_rows': store.n, 'anchors': len(anchors),
            'selected_patterns': len(selected), 'families': dict(Counter(p['predicate']['type'] for p in selected)),
            'raw_templates': universe['raw_templates_including_certified_rejections'],
            'exactly_pruned_pairs': full_pairs - evaluated_pairs, 'physically_evaluated_pairs': evaluated_pairs,
            'elapsed_seconds': time.perf_counter() - started,
            'files': {name: file_sha(directory / name) for name in
                ('protocol.json', 'anchors.json', 'raw_universe.json', 'screening.sqlite', 'pattern_specs.json')}}
        immutable_json(done, result)
        atomic_json(directory / 'progress.json', result)
        return result
    finally:
        if db is not None:
            db.close()
