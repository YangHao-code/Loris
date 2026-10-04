"""Complete fresh lexical/NLP anchor extraction, without accuracy screening.

All current-corpus TF-IDF 1-2grams, noun chunks, verb/AUX lemmas, named entities,
their observed morphological forms, and declared dictionaries are retained.
The PDF does not specify these tokenizer/model/default choices; the executable
protocol records them explicitly. This module never reads labels or old banks.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np


SCHEMA = 'paper36_fresh_full_anchor_field_v1'
TFIDF_TOKEN_PATTERN = r'(?u)\b\w+\b'
TOKEN = re.compile(TFIDF_TOKEN_PATTERN)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_surface(text):
    # ASCII lowercasing preserves literal spans under IGNORECASE. Do not turn
    # an observed Turkish İ into an unobserved two-codepoint i+combining-dot.
    return text.lower() if text.isascii() else text


def ordered_forms(forms):
    return tuple(sorted(set(forms), key=lambda value: (-len(value), value)))


def word_alternatives_regex(alternatives):
    """Canonical structure shared with the exact inverted-index evaluator."""
    alternatives = [ordered_forms(forms) for forms in alternatives]
    if not alternatives or any(not forms or any(re.fullmatch(r'(?u)\w+', form) is None for form in forms)
                               for forms in alternatives):
        raise ValueError('literal token structure requires nonempty complete word alternatives')
    raw = r'\s+'.join(r'\b(?:' + '|'.join(re.escape(form) for form in forms) + r')\b'
                       for forms in alternatives)
    return raw, [list(forms) for forms in alternatives]


def _literal_part(text):
    raw = re.escape(text)
    if text and re.match(r'(?u)\w', text[0]):
        raw = r'\b' + raw
    if text and re.match(r'(?u)\w', text[-1]):
        raw += r'\b'
    return raw


def literal_regex(text):
    text = normalize_surface(text)
    pieces = re.split(r'(\s+)', text)
    words = [piece for piece in pieces if piece and not piece.isspace()]
    if words and all(re.fullmatch(r'(?u)\w+', word) for word in words):
        return word_alternatives_regex([[word] for word in words])
    # Preserve punctuation and arbitrary-length spans instead of dropping
    # them to obtain the easier indexed case.
    raw = ''.join(r'\s+' if piece.isspace() else re.escape(piece) for piece in pieces if piece)
    if text and re.match(r'(?u)\w', text[0]): raw = r'\b' + raw
    if text and re.match(r'(?u)\w', text[-1]): raw += r'\b'
    return raw, None


def lemma_key(token):
    return canonical([normalize_surface(token[1] or token[0]), token[2]])


def morphological_regex(text, tokens, lexicon):
    if not tokens:
        raise ValueError('morphological anchor requires observed tokens')
    alternatives = [ordered_forms(lexicon.get(lemma_key(token), [normalize_surface(token[0])])) for token in tokens]
    gaps = [text[left[4]:right[3]] for left, right in zip(tokens, tokens[1:])]
    if all(gap and gap.isspace() for gap in gaps) and all(
            all(re.fullmatch(r'(?u)\w+', value) for value in forms) for forms in alternatives):
        return word_alternatives_regex(alternatives)
    expressions = ['(?:' + '|'.join(_literal_part(value) for value in forms) + ')' for forms in alternatives]
    raw = expressions[0]
    for gap, expression in zip(gaps, expressions[1:]):
        raw += (r'\s+' if gap and gap.isspace() else re.escape(gap)) + expression
    return raw, None


def spacy_record(document, *, row, attr, identity):
    """Every raw token/span survives; no POS/entity type, length or DF cap."""
    text = document.text
    if text and len(document) and not all(document.has_annotation(name) for name in ('DEP', 'POS', 'LEMMA', 'ENT_IOB')):
        raise ValueError('complete parser/POS/lemma/NER annotations required')
    tokens = [[token.text, token.lemma_, token.pos_, token.idx, token.idx + len(token)] for token in document]
    noun_chunks = [[span.start, span.end, span.start_char, span.end_char] for span in document.noun_chunks]
    entities = [[span.start, span.end, span.start_char, span.end_char, span.label_] for span in document.ents]
    verbs = [token.i for token in document if token.pos_ in ('VERB', 'AUX')]
    for surface, lemma, pos, start, end in tokens:
        if text[start:end] != surface:
            raise ValueError('token/source span mismatch')
    return {'row': int(row), 'attr': attr, 'identity': identity, 'text': text,
        'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'tokens': tokens,
        'noun_chunks': noun_chunks, 'verbs': verbs, 'entities': entities}


def read_records(paths):
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)


def observed_lexicon(records):
    forms = defaultdict(set)
    for record in records:
        for token in record['tokens']:
            forms[lemma_key(token)].add(normalize_surface(token[0]))
    return {key: list(ordered_forms(values)) for key, values in sorted(forms.items())}


def tfidf_occurrences(text):
    """Default lowercased word features with lossless ORIGINAL source offsets."""
    lowered = text.lower()
    # Full-string lower can change character shape by context; individual-char
    # lower lengths still give the offset map, including expanding Unicode İ.
    source = [index for index, char in enumerate(text) for _ in char.lower()]
    if len(source) != len(lowered):
        raise ValueError('lowercase character alignment requires explicit handling')
    tokens = [(match.group(), source[match.start()], source[match.end() - 1] + 1)
              for match in TOKEN.finditer(lowered)]
    for index, token in enumerate(tokens):
        yield token[0], token[1], token[2], [[token[1], token[2]]]
        if index + 1 < len(tokens):
            right = tokens[index + 1]
            yield token[0] + ' ' + right[0], token[1], right[2], [[token[1], token[2]], [right[1], right[2]]]


class AnchorStore:
    """Compact anchor identities with complete external per-occurrence evidence."""
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE anchors(key TEXT PRIMARY KEY,raw TEXT NOT NULL,flags INTEGER NOT NULL,literal_json TEXT)')
        self.db.execute('CREATE TABLE origins(origin_id TEXT PRIMARY KEY,anchor_key TEXT NOT NULL,kind TEXT NOT NULL,detail_json TEXT NOT NULL)')
        self.db.execute('CREATE TABLE evidence(origin_id TEXT NOT NULL,row INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,span TEXT NOT NULL,token_spans_json TEXT NOT NULL,regex_matches_span INTEGER NOT NULL)')
        self.anchors, self.origins, self.compiled, self.counts = {}, {}, {}, Counter()
        self.buffer = []

    def register(self, raw, flags, structure, kind, detail):
        key = digest({'raw': raw, 'flags': int(flags)})
        if key not in self.anchors:
            self.anchors[key] = {'key': key, 'raw': raw, 'flags': int(flags),
                'literal_alternatives_by_token': structure, 'origins': []}
            self.db.execute('INSERT INTO anchors VALUES(?,?,?,?)',
                (key, raw, int(flags), canonical(structure) if structure is not None else None))
        elif structure is not None and self.anchors[key]['literal_alternatives_by_token'] is None:
            self.anchors[key]['literal_alternatives_by_token'] = structure
            self.db.execute('UPDATE anchors SET literal_json=? WHERE key=?', (canonical(structure), key))
        elif structure is not None and self.anchors[key]['literal_alternatives_by_token'] != structure:
            raise ValueError('same regex cannot have inconsistent literal token structure')
        origin_id = digest({'anchor_key': key, 'kind': kind, 'detail': detail})
        if origin_id not in self.origins:
            origin = {'origin_id': origin_id, 'kind': kind, 'detail': detail,
                'evidence_database': 'origins.sqlite', 'evidence_lookup': 'evidence.origin_id'}
            self.origins[origin_id] = origin
            self.anchors[key]['origins'].append(origin)
            self.db.execute('INSERT INTO origins VALUES(?,?,?,?)', (origin_id, key, kind, canonical(detail)))
        return key, origin_id

    def occurrence(self, key, origin_id, record, start, end, token_spans=()):
        text = record['text']
        if not 0 <= start <= end <= len(text):
            raise ValueError('anchor evidence span outside original document')
        if key not in self.compiled:
            item = self.anchors[key]
            self.compiled[key] = re.compile(item['raw'], item['flags'])
        matched = self.compiled[key].match(text, start)
        matches = matched is not None and matched.end() == end
        self.buffer.append((origin_id, record['row'], start, end, text[start:end], canonical(token_spans), int(matches)))
        self.counts[origin_id] += 1
        if len(self.buffer) >= 8192:
            self.flush()

    def flush(self):
        if self.buffer:
            self.db.executemany('INSERT INTO evidence VALUES(?,?,?,?,?,?,?)', self.buffer)
            self.buffer.clear()
        self.db.commit()

    def finish(self):
        self.flush()
        self.db.execute('CREATE INDEX evidence_by_origin ON evidence(origin_id)')
        self.db.execute('CREATE INDEX evidence_by_row ON evidence(row)')
        self.db.commit()
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
        for origin_id, origin in self.origins.items():
            origin['evidence_occurrences'] = self.counts[origin_id]
        return sorted(self.anchors.values(), key=lambda item: item['key'])


def add_nlp_anchors(store, record, lexicon):
    text, tokens = record['text'], record['tokens']
    spans = [('noun_chunk', part[:4], {}) for part in record['noun_chunks']]
    spans += [('named_entity', part[:4], {'entity_type': part[4]}) for part in record['entities']]
    for kind, (first, last, start, end), detail in spans:
        surface = text[start:end]
        raw, structure = literal_regex(surface)
        key, origin = store.register(raw, re.IGNORECASE, structure, kind + '_literal', dict(detail, surface=surface))
        store.occurrence(key, origin, record, start, end)
        raw, structure = morphological_regex(text, tokens[first:last], lexicon)
        lemma_sequence = [lemma_key(token) for token in tokens[first:last]]
        key, origin = store.register(raw, re.IGNORECASE, structure, kind + '_morph', dict(detail, lemmas=lemma_sequence))
        store.occurrence(key, origin, record, start, end)
    for index in record['verbs']:
        token = tokens[index]
        raw, structure = morphological_regex(text, [token], lexicon)
        key, origin = store.register(raw, re.IGNORECASE, structure, 'verb_lemma', {'lemma_key': lemma_key(token)})
        store.occurrence(key, origin, record, token[3], token[4])


def build_field(records, lexicon, dictionaries, output, *, cluster, attr, protocol_id):
    """Extract one complete field; filtering and pairs are intentionally absent."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    store = AnchorStore(output / 'origins.sqlite')
    records = list(records)
    texts = [record['text'] for record in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=TFIDF_TOKEN_PATTERN,
        lowercase=True, stop_words=None, min_df=1, max_df=1., max_features=None)
    if any(TOKEN.search(text.lower()) for text in texts):
        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out().tolist()
        idf = vectorizer.idf_.tolist()
        save_npz(str(output / 'tfidf.npz'), matrix)
        references = {}
        for index, feature in enumerate(names):
            raw, structure = word_alternatives_regex([[word] for word in feature.split(' ')])
            references[feature] = store.register(raw, re.IGNORECASE, structure,
                'tfidf_unigram' if ' ' not in feature else 'tfidf_bigram',
                {'feature': feature, 'feature_index': index, 'idf': idf[index]})
    else:
        names, idf, references = [], [], {}
        matrix = None
    (output / 'tfidf_vocabulary.json').write_text(canonical({'features': names, 'idf': idf,
        'document_rows': [record['row'] for record in records], 'min_df': 1, 'max_df': 1.,
        'max_features': None, 'top_k': None, 'token_pattern': TFIDF_TOKEN_PATTERN,
        'stop_words': None, 'ngram_range': [1, 2], 'lowercase': True}))
    dictionary_refs = []
    for entry in dictionaries:
        key, origin = store.register(entry['raw'], entry['flags'], None, 'dictionary_regex',
            {key: value for key, value in entry.items() if key not in ('raw', 'flags')})
        dictionary_refs.append((key, origin, re.compile(entry['raw'], entry['flags'])))
    for record in records:
        for feature, start, end, token_spans in tfidf_occurrences(record['text']):
            key, origin = references[feature]
            store.occurrence(key, origin, record, start, end, token_spans)
        add_nlp_anchors(store, record, lexicon)
        for key, origin, compiled in dictionary_refs:
            for match in compiled.finditer(record['text']):
                store.occurrence(key, origin, record, match.start(), match.end())
    anchors = store.finish()
    provenance = {'schema': SCHEMA, 'protocol_id': protocol_id, 'cluster': int(cluster), 'attr': attr,
        'fit_rows': len(records), 'document_rows': [record['row'] for record in records],
        'tfidf_features': len(names), 'tfidf_doc_feature_pairs': int(matrix.nnz) if matrix is not None else 0,
        'anchors': len(anchors), 'origins': len(store.origins), 'evidence_occurrences': sum(store.counts.values()),
        'dictionary_regexes_retained': len(dictionaries),
        'literal_structured_anchors': sum(a['literal_alternatives_by_token'] is not None for a in anchors),
        'no_min_df_or_topk_screening': True, 'no_pairs_generated_or_filtered': True,
        'no_labels_or_predictions_used': True, 'all_observed_morphological_forms_retained': True,
        'TFIDF_token_adjacent_bigrams_can_cross_punctuation_regex_requires_whitespace': True,
        'source_span_regex_fullmatch_recorded_separately': True}
    (output / 'anchors.json').write_text(canonical({'anchors': anchors, 'provenance': provenance}))
    return provenance
