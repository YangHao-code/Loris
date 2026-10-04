"""Finite, fit-text-derived PDF36 pattern domain; no label/coverage pruning.

This is an explicit extractor protocol, not an assertion that PDF36 specifies
spaCy or this vocabulary. Every declared instance is retained. Dependency
edges propose pairs of complete phrases, while evaluation still uses the
original document-wide predicates (in particular, global first-match Before).
The canonical pattern F1 selector is unchanged.
"""
import gzip
import importlib
import json
import operator
from pathlib import Path
import re
import sqlite3
import time
import zlib

import numpy as np

from loris._engine.pattern_anchors import literal_regex
from loris._engine.pattern_discovery import CompleteRegexStore
from loris._engine.rlearner import array_identity, canonical, digest
from loris._engine.rlearner_config import PatternSpec, ParameterDomain, FREQUENCY_OPS
from loris.predicates import MatchPredicate, FreqPredicate, CooccurPredicate, BeforePredicate
from .artifacts import bound_path, file_record, write_json
from .patterns import FitLabelAssigner, PatternCandidate, PatternSelectionProtocol, select_patterns


SCHEMA = 'paper36_semantic_raw_pattern_domain_v1'
ATTRS = ('cnt', 'ttl', 'mtd')
CONTENT_POS = ('NOUN', 'PROPN', 'VERB', 'ADJ')
OPS = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
       '<=': operator.le, '>': operator.gt, '>=': operator.ge}


def _read(path):
    return json.loads(Path(path).read_text())


def _readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def _sources():
    modules = ('loris._engine.pattern_anchors', 'loris._engine.pattern_discovery',
               'loris._engine.regex_superset_index', 'loris._engine.rlearner_config',
               'loris.predicates._core', 'loris._learning.patterns')
    return [file_record(__file__)] + [file_record(importlib.import_module(m).__file__) for m in modules]


def _integer_axis(values, name):
    """Validate before conversion so row identities cannot be truncated/wrapped."""
    axis = np.asarray(values)
    if (axis.ndim != 1 or axis.dtype.kind not in 'iu'
            or any(isinstance(value, (bool, np.bool_)) for value in values)
            or np.any(axis < 0) or np.any(axis > np.iinfo(np.int64).max)):
        raise ValueError(name + ' requires nonnegative integer values within int64 range')
    return axis.astype(np.int64, copy=False)


def _axes(context, assignments):
    n = len(context['docs'])
    fit = _integer_axis(context['fit_indices'], 'fit row axis')
    control = _integer_axis(context['receiver_indices'], 'control row axis')
    rows = list(context['row_ids'])
    roles = list(context['row_roles'])
    clusters = _integer_axis(assignments, 'cluster assignment axis')
    if (len(rows) != n or len(roles) != n or len(set(rows)) != n
            or fit.ndim != 1 or control.ndim != 1
            or len(fit) == 0 or len(control) == 0
            or sorted(np.concatenate((fit, control)).tolist()) != list(range(n))
            or clusters.shape != (len(fit),) or clusters.dtype.kind not in 'iu'
            or np.any(clusters < 0)
            or any(roles[i] not in ('fit', 'source_fit') for i in fit)
            or any(roles[i] not in ('control', 'val_ctrl') for i in control)):
        raise ValueError('complete disjoint fit/control axes and integer fit-only clusters required; no test rows')
    presence = {}
    text_ids = {}
    for attr in ATTRS:
        texts = [d.get_attr(attr) for d in context['docs']]
        if any(not isinstance(text, str) for text in texts):
            raise ValueError('complete original string attributes required')
        p = np.asarray(context.get('attribute_presence', {}).get(attr, [bool(t) for t in texts]))
        if p.shape != (n,) or not np.isin(p, [0, 1]).all():
            raise ValueError('complete binary attribute presence required')
        presence[attr] = p.astype(bool)
        text_ids[attr] = digest(texts)
    binding = {'rows': rows, 'row_roles': roles, 'fit_indices': fit.tolist(),
               'control_indices': control.tolist(), 'cluster_assignments': clusters.tolist(),
               'texts': text_ids, 'presence': {a: array_identity(p) for a, p in presence.items()}}
    return fit, control, clusters.astype(np.int64), presence, binding


def semantic_record(doc):
    """Complete parse plus span-backed anchors/pairs, independent of labels."""
    if len(doc) and not all(doc.has_annotation(k) for k in ('DEP', 'POS', 'LEMMA', 'ENT_IOB')):
        raise ValueError('POS/dependency/lemma/entity annotations required')
    spans = {}

    def add(span, kind):
        if span.text.strip() and any(c.isalpha() for c in span.text):
            key = (span.start, span.end)
            spans.setdefault(key, {'start': span.start_char, 'end': span.end_char,
                'token_start': span.start, 'token_end': span.end, 'root': span.root.i,
                'surface': span.text, 'kinds': []})['kinds'].append(kind)

    for token in doc:
        if token.pos_ in CONTENT_POS and token.is_alpha:
            add(doc[token.i:token.i + 1], 'content_' + token.pos_)
    for chunk in doc.noun_chunks:
        add(chunk, 'noun_phrase')
    for ent in doc.ents:
        add(ent, 'entity_' + ent.label_)
    # Each token projects to a complete entity first, then complete noun phrase,
    # then its content token. This avoids substituting fragments of entity names.
    projected = {}
    for token in doc:
        choices = [s for s in spans.values() if s['token_start'] <= token.i < s['token_end']]
        if choices:
            projected[token.i] = min(choices, key=lambda s: (
                0 if any(k.startswith('entity_') for k in s['kinds']) else
                1 if 'noun_phrase' in s['kinds'] else 2,
                -(s['token_end'] - s['token_start']), s['token_start']))
    pairs = {}

    def pair(left, right, kind):
        if left['end'] > right['start']:
            left, right = right, left
        if left['end'] > right['start']:
            return
        key = (left['token_start'], left['token_end'], right['token_start'], right['token_end'])
        pairs.setdefault(key, {'left': [left['token_start'], left['token_end']],
            'right': [right['token_start'], right['token_end']], 'kinds': []})['kinds'].append(kind)

    for token in doc:
        if (token.i != token.head.i and token.i in projected and token.head.i in projected
                and token.sent.start == token.head.sent.start):
            pair(projected[token.i], projected[token.head.i], 'dependency_' + token.dep_)
    for sentence in doc.sents:
        phrases = [s for s in spans.values() if sentence.start <= s['token_start'] < sentence.end
                   and ('noun_phrase' in s['kinds'] or any(k.startswith('entity_') for k in s['kinds']))]
        # Maximal nonoverlapping phrases; entities within noun chunks are
        # retained as anchors but do not create artificial nested adjacency.
        ordered, end = [], -1
        for span in sorted(phrases, key=lambda s: (s['token_start'], -s['token_end'])):
            if span['token_start'] >= end:
                ordered.append(span)
                end = span['token_end']
        for left, right in zip(ordered, ordered[1:]):
            pair(left, right, 'adjacent_phrase')
    return {'text': doc.text, 'tokens': [[t.text, t.lemma_, t.pos_, t.dep_, t.head.i,
            t.idx, t.idx + len(t)] for t in doc],
        'sentences': [[s.start, s.end] for s in doc.sents],
        'anchors': sorted(spans.values(), key=lambda s: (s['start'], s['end'])),
        'pairs': list(pairs.values())}


def extract_semantic_domain(context, cluster_assignments, *, output_dir,
                            input_binding, nlp_model='en_core_web_sm'):
    """Parse every FIT text, then count the entire finite domain; no gold read."""
    import spacy
    start = time.monotonic()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    fit, control, clusters, presence, axes = _axes(context, cluster_assignments)
    nlp = spacy.load(nlp_model)
    package = importlib.import_module(nlp_model)
    model_root = Path(package.__file__).resolve().parent
    model_files = [file_record(p) for p in sorted(model_root.rglob('*'))
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    policy = {'schema': SCHEMA, 'sources': _sources(), 'input_binding': input_binding,
        'axes': axes, 'spacy_version': spacy.__version__, 'model': nlp_model,
        'model_meta': nlp.meta, 'model_files': model_files,
        'anchor_domain': 'all alphabetic NOUN/PROPN/VERB/ADJ tokens, complete noun phrases and named entities containing letters',
        'pair_domain': 'observed same-sentence dependency edges projected to complete phrases, or adjacent maximal nonoverlapping noun/entity phrases',
        'frequency_domain': 'all integer eta from 0 through max fit occurrence count + 1, all six comparisons',
        'predicate_semantics': 'original entire attribute, global first occurrence for Before; missing attributes masked false',
        'order': 'Match then Cooccur then Before then Freq; within family lexicographic stable anchor/pair keys; Freq all six declared ops then ascending integer eta',
        'raw_unfiltered_vocabulary': True, 'gold_used_for_extraction': False,
        'posthoc_cap': None, 'minimum_df': None, 'control_text_used_for_vocabulary': False}
    write_json(directory / 'PROTOCOL.json', policy, exclusive=True)
    db = sqlite3.connect(str(directory / 'domain.sqlite'))
    db.executescript('CREATE TABLE anchors (cluster INTEGER,key TEXT,attr TEXT,definition TEXT,max_fit_count INTEGER, PRIMARY KEY(cluster,key));'
        'CREATE TABLE pairs (cluster INTEGER,family TEXT,left_key TEXT,right_key TEXT,PRIMARY KEY(cluster,family,left_key,right_key));')
    parse_count = 0
    rows = [(int(i), int(cl), attr) for i, cl in zip(fit, clusters) for attr in ATTRS if presence[attr][i]]
    longest = max([len(context['docs'][i].get_attr(a)) for i, _, a in rows] + [0])
    nlp.max_length = max(nlp.max_length, longest + 1)
    with gzip.open(directory / 'parsed.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for (row, cluster, attr), parsed in zip(rows, nlp.pipe(
                (context['docs'][i].get_attr(a) for i, _, a in rows), batch_size=32, n_process=1)):
            record = semantic_record(parsed)
            record.update(row=row, row_id=axes['rows'][row], cluster=cluster, attr=attr)
            keys = {}
            for anchor in record['anchors']:
                raw, structure = literal_regex(anchor['surface'])
                definition = {'raw': raw, 'flags': int(re.IGNORECASE),
                              'literal_alternatives_by_token': structure}
                key = digest([attr, raw, int(re.IGNORECASE)])
                anchor['key'] = key
                keys[(anchor['token_start'], anchor['token_end'])] = key
                db.execute('INSERT OR IGNORE INTO anchors VALUES (?,?,?,?,NULL)',
                           (cluster, key, attr, canonical(definition)))
            for pair in record['pairs']:
                left, right = keys[tuple(pair['left'])], keys[tuple(pair['right'])]
                if left == right:
                    continue
                lo, hi = sorted((left, right))
                db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?)', (cluster, 'Cooccur', lo, hi))
                db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?)', (cluster, 'Before', left, right))
            stream.write(canonical(record) + '\n')
            parse_count += 1
            if parse_count % 100 == 0:
                db.commit()
                write_json(directory / 'progress.json', {'stage': 'parse', 'parsed': parse_count,
                    'expected_parses': len(rows), 'elapsed_seconds': time.monotonic() - start})
    db.commit()
    # Complete counts/first positions are reusable across clusters and Freq
    # instances. Control text contributes masks only, never extraction decisions.
    store = CompleteRegexStore(context['docs'], directory / 'exact_regex', presence, regex_index='literal')
    summary = {}
    try:
        for cluster in sorted(set(clusters.tolist())):
            fit_rows = fit[clusters == cluster]
            count = 0
            for key, attr, definition in db.execute('SELECT key,attr,definition FROM anchors WHERE cluster=? ORDER BY key', (cluster,)).fetchall():
                stats = store.counts_first(attr, json.loads(definition))
                maximum = int(np.max(stats[0, fit_rows]))
                db.execute('UPDATE anchors SET max_fit_count=? WHERE cluster=? AND key=?', (maximum, cluster, key))
                count += 1 + len(FREQUENCY_OPS) * (maximum + 2)
            families = {'Match': db.execute('SELECT COUNT(*) FROM anchors WHERE cluster=?', (cluster,)).fetchone()[0]}
            families['Freq'] = count - families['Match']
            for family in ('Cooccur', 'Before'):
                families[family] = db.execute('SELECT COUNT(*) FROM pairs WHERE cluster=? AND family=?', (cluster, family)).fetchone()[0]
            summary[str(cluster)] = {'fit_indices': fit_rows.tolist(), 'fit_row_ids': [axes['rows'][i] for i in fit_rows],
                'control_indices': control.tolist(), 'control_row_ids': [axes['rows'][i] for i in control],
                'families': families, 'expected_candidates': sum(families.values())}
            db.commit()
            store.commit()
            write_json(directory / 'progress.json', {'stage': 'count', 'clusters': summary,
                'elapsed_seconds': time.monotonic() - start})
        stats = dict(store.stats)
    finally:
        store.close()
        db.close()
    for source in policy['sources']:
        bound_path(source)
    result = {'schema': SCHEMA, 'protocol': file_record(directory / 'PROTOCOL.json'),
        'domain_database': file_record(directory / 'domain.sqlite'),
        'parsed_records': file_record(directory / 'parsed.jsonl.gz'), 'clusters': summary,
        'expected_candidates': sum(s['expected_candidates'] for s in summary.values()),
        'parsed_attributes': parse_count, 'complete_parse': parse_count == len(rows),
        'regex_stats': stats, 'elapsed_seconds': time.monotonic() - start}
    write_json(directory / 'DOMAIN.json', result, exclusive=True)
    return result


def _domain(directory):
    directory = Path(directory).resolve()
    result = _read(directory / 'DOMAIN.json')
    policy = _read(bound_path(result['protocol']))
    if result.get('schema') != SCHEMA or policy.get('schema') != SCHEMA or not result.get('complete_parse'):
        raise ValueError('complete semantic raw domain required')
    for record in (result['domain_database'], result['parsed_records']) + tuple(policy['sources']):
        bound_path(record)
    return result, policy


def _spec(family, attr, left, right=None, op=None, eta=None):
    l = re.compile(left['raw'], left['flags'])
    if family == 'Match':
        predicate = MatchPredicate(attr, l)
    elif family == 'Freq':
        predicate = FreqPredicate(attr, l, op, eta)
    else:
        predicate = (CooccurPredicate if family == 'Cooccur' else BeforePredicate)(attr,
            l, re.compile(right['raw'], right['flags']))
    key = family + ':' + digest([attr, left, right, op, eta])
    return PatternSpec(key, predicate, ParameterDomain(values=(eta,), integer=True) if family == 'Freq' else None,
                       (op,) if family == 'Freq' else FREQUENCY_OPS)


def _pack(mask):
    bits = np.packbits(np.asarray(mask, bool), bitorder='little').tobytes()
    import hashlib
    return zlib.compress(bits), hashlib.sha256(bits).hexdigest()


def _unpack(blob, expected, n):
    import hashlib
    bits = zlib.decompress(blob)
    if hashlib.sha256(bits).hexdigest() != expected or len(bits) != (n + 7) // 8:
        raise ValueError('raw semantic mask checksum/length differs')
    return np.unpackbits(np.frombuffer(bits, np.uint8), bitorder='little')[:n].astype(bool)


def materialize_semantic_masks(context, domain_dir, *, output_dir):
    """Save EVERY raw fit/control mask, before any label assignment or F1."""
    start = time.monotonic()
    domain, policy = _domain(domain_dir)
    fit, control, clusters, presence, axes = _axes(context, policy['axes']['cluster_assignments'])
    if axes != policy['axes']:
        raise ValueError('raw domain text/row/presence binding differs')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    db = _readonly(bound_path(domain['domain_database']))
    store = CompleteRegexStore(context['docs'], Path(domain_dir) / 'exact_regex', presence, regex_index='literal')
    output = sqlite3.connect(str(directory / 'raw_masks.sqlite'))
    output.execute('CREATE TABLE raw (cluster INTEGER,ordinal INTEGER,key TEXT,spec TEXT,fit_mask BLOB,fit_sha TEXT,control_mask BLOB,control_sha TEXT, PRIMARY KEY(cluster,ordinal), UNIQUE(cluster,key))')
    count = 0
    try:
        for cluster, detail in sorted(domain['clusters'].items(), key=lambda x: int(x[0])):
            cluster = int(cluster)
            fi, ci = np.asarray(detail['fit_indices']), np.asarray(detail['control_indices'])
            anchors = {key: (attr, json.loads(definition), maximum) for key, attr, definition, maximum in
                       db.execute('SELECT key,attr,definition,max_fit_count FROM anchors WHERE cluster=? ORDER BY key', (cluster,))}
            ordinal = 0

            def save(spec, mask):
                nonlocal ordinal, count
                f, fs = _pack(mask[fi])
                c, cs = _pack(mask[ci])
                output.execute('INSERT INTO raw VALUES (?,?,?,?,?,?,?,?)',
                    (cluster, ordinal, spec.key, canonical(spec.to_dict()), f, fs, c, cs))
                ordinal += 1
                count += 1
                if count % 1000 == 0:
                    output.commit()
                    write_json(directory / 'progress.json', {'raw_masks': count,
                        'expected_candidates': domain['expected_candidates'], 'elapsed_seconds': time.monotonic() - start})

            for family in ('Match',):
                for key, (attr, anchor, maximum) in anchors.items():
                    values = store.counts_first(attr, anchor)[0]
                    if family == 'Match':
                        save(_spec(family, attr, anchor), (values > 0) & presence[attr])
            for family in ('Cooccur', 'Before'):
                for left, right in db.execute('SELECT left_key,right_key FROM pairs WHERE cluster=? AND family=? ORDER BY left_key,right_key', (cluster, family)):
                    attr, a, _ = anchors[left]
                    rattr, b, _ = anchors[right]
                    if attr != rattr:
                        raise ValueError('pair attribute mismatch')
                    x, y = store.counts_first(attr, a), store.counts_first(attr, b)
                    mask = (x[0] > 0) & (y[0] > 0) & presence[attr]
                    if family == 'Before':
                        mask &= x[1] < y[1]
                    save(_spec(family, attr, a, b), mask)
            for key, (attr, anchor, maximum) in anchors.items():
                values = store.counts_first(attr, anchor)[0]
                for op in FREQUENCY_OPS:
                    for eta in range(maximum + 2):
                        save(_spec('Freq', attr, anchor, op=op, eta=eta), OPS[op](values, eta) & presence[attr])
            if ordinal != detail['expected_candidates']:
                raise ValueError('raw instance count differs from complete domain')
            output.commit()
        stats = dict(store.stats)
    finally:
        db.close()
        store.close()
        output.close()
    _domain(domain_dir)
    receipt = {'schema': SCHEMA + '_masks', 'domain': file_record(Path(domain_dir) / 'DOMAIN.json'),
        'database': file_record(directory / 'raw_masks.sqlite'), 'clusters': domain['clusters'],
        'raw_candidates': count, 'expected_candidates': domain['expected_candidates'],
        'all_raw_masks_saved': count == domain['expected_candidates'],
        'label_or_control_gold_read': False, 'regex_stats': stats,
        'elapsed_seconds': time.monotonic() - start}
    write_json(directory / 'COMPLETE.json', receipt, exclusive=True)
    return receipt


def iter_semantic_candidates(raw_dir, cluster_id, *, fit_truth, fit_baseline_predictions):
    """Reassign labels from saved FIT masks; never reuse an old selected Pi."""
    raw = _read(Path(raw_dir) / 'COMPLETE.json')
    if raw.get('schema') != SCHEMA + '_masks' or not raw.get('all_raw_masks_saved'):
        raise ValueError('complete raw mask archive required')
    domain_record = bound_path(raw['domain'])
    domain, policy = _domain(domain_record.parent)
    if raw['clusters'] != domain['clusters'] or raw['raw_candidates'] != domain['expected_candidates']:
        raise ValueError('raw mask/domain count binding differs')
    detail = raw['clusters'][str(cluster_id)]
    assign = FitLabelAssigner(fit_truth, row_ids=detail['fit_row_ids'], baseline_predictions=fit_baseline_predictions)
    if len(assign.y) != len(detail['fit_row_ids']):
        raise ValueError('complete cluster fit rows required')
    db = _readonly(bound_path(raw['database']))
    count = 0
    try:
        for ordinal, key, spec, fm, fs, cm, cs in db.execute('SELECT ordinal,key,spec,fit_mask,fit_sha,control_mask,control_sha FROM raw WHERE cluster=? ORDER BY ordinal', (int(cluster_id),)):
            if ordinal != count:
                raise ValueError('incomplete raw ordinal axis')
            fit_mask = _unpack(fm, fs, len(detail['fit_row_ids']))
            control_mask = _unpack(cm, cs, len(detail['control_row_ids']))
            labels, provenance = assign(fit_mask)
            provenance.update(raw_extraction={'schema': SCHEMA, 'raw_unfiltered_vocabulary': True,
                'domain_sha256': raw['domain']['sha256'], 'raw_database_sha256': raw['database']['sha256'],
                'cluster': int(cluster_id)}, frequency_instance_exact=True,
                legacy_coverage_entropy_prefilter=False)
            yield PatternCandidate(key, json.loads(spec), control_mask, labels, provenance)
            count += 1
        if count != detail['expected_candidates']:
            raise ValueError('complete raw candidate count differs')
    finally:
        db.close()


def select_semantic_patterns(raw_dir, cluster_id, *, fit_truth, fit_baseline_predictions,
                             control_truth, control_baseline_predictions, labels,
                             input_binding, output_dir, strategy='greedy_marginal'):
    """Thin adapter: original full-domain marginal-F1 selection, unchanged."""
    raw = _read(Path(raw_dir) / 'COMPLETE.json')
    detail = raw['clusters'][str(cluster_id)]
    binding = dict(input_binding, role='control', expected_rows=len(detail['control_row_ids']),
                   semantic_raw_masks=file_record(Path(raw_dir) / 'COMPLETE.json'))
    return select_patterns(iter_semantic_candidates(raw_dir, cluster_id,
        fit_truth=fit_truth, fit_baseline_predictions=fit_baseline_predictions),
        truth=control_truth, baseline_predictions=control_baseline_predictions,
        row_ids=detail['control_row_ids'], labels=labels, input_binding=binding,
        protocol=PatternSelectionProtocol(expected_candidates=detail['expected_candidates'], strategy=strategy),
        output_dir=output_dir)
