"""Explicit outer-loop revisions and complete conflict evidence.

No rule, model score, or signed fact is repaired inside the Chase. A revision
is a new authenticated input version, evaluated from its declared roots.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from loris._engine.typed_session import TypedChaseSession, complete_state_identity, seed_identity
from loris._engine.rdl_reference import prepare_attributes
from .artifacts import bound_path, digest, file_record, plain


def root_keys(gamma):
    keys = {'%s:%d:%d' % (sign, i, j)
        for sign, facts in (('pos', gamma.pos), ('neg', gamma.neg)) for i, j in np.argwhere(facts)}
    for kind in ('subset', 'strict_subset'):
        keys.update('%s:%d:%d' % (kind, a, b) for a, b in getattr(gamma, kind))
    return keys


def corpus_identity(context):
    """Bind every ordered document and attribute, independently of Gamma/theta."""
    docs = tuple(context['docs'])
    values, presence = prepare_attributes(docs, context.get('attribute_values'), context.get('attribute_presence'))
    axes = {}
    for key in ('row_ids', 'identities'):
        if key in context:
            axis = plain(context[key])
            if not isinstance(axis, list) or len(axis) != len(docs) or len({digest(row) for row in axis}) != len(docs):
                raise ValueError('corpus identity requires a complete unique ordered ' + key + ' axis')
            axes[key] = axis
    body = {'schema': 'paper36_immutable_corpus_identity_v1', 'rows': len(docs),
        'row_identifiers': axes,
        'ordered_docs_sha256': digest([{'cnt': d.cnt, 'ttl': d.ttl, 'mtd': d.mtd} for d in docs]),
        'attribute_values_sha256': {key: digest(column) for key, column in sorted(values.items())},
        'attribute_presence_sha256': {key: digest(mask) for key, mask in sorted(presence.items())}}
    return dict(body, sha256=digest(body))


def static_model_identity(context):
    """Static ML decisions are a separately reviewed model interpretation."""
    result = {}
    shape = (len(context['docs']), len(context['labels']))
    for alias, values in context.get('model_decisions', {}).items():
        values = np.asarray(values)
        if values.shape != shape or not np.isin(values, (0, 1)).all():
            raise ValueError('static model decisions require complete binary corpus/label axes')
        result[alias] = {'shape': list(shape), 'sha256': digest(values.astype(bool))}
    return result


def conflict_certificate(result, *, session, seed, root_sources, stage):
    """Keep both actual proof branches; never blame the last firing rule."""
    if result.status != 'conflict':
        raise ValueError('a failed complete Chase is required for Conflict(K)')
    proofs, collisions = {}, []
    def retain(key):
        pending = [key]
        while pending:
            current = pending.pop()
            if current in proofs:
                continue
            if current not in result.proof_dag:
                raise ValueError('conflicting fact has no authentic proof: ' + current)
            node = plain(result.proof_dag[current])
            proofs[current] = node
            pending.extend(node.get('dependencies', ()))
        return key
    for row, label in np.argwhere(result.positive_facts & result.negative_facts):
        row, label = int(row), int(label)
        collisions.append({'document_index': row, 'label_index': label,
            'label': session.labels[label], 'positive_source': retain('pos:%d:%d' % (row, label)),
            'negative_source': retain('neg:%d:%d' % (row, label))})
    # Comparison infeasibility may have no signed collision. Retain all its
    # premises rather than inventing one uniquely culpable rule or witness.
    if result.comparison_conflicts:
        for key in result.proof_dag:
            retain(key)
    body = {'schema': 'paper36_outer_conflict_certificate_v1', 'outcome': 'Conflict',
        'stage': stage, 'Sigma': session.program_id, 'theta': session.snapshot_id,
        'context_id': session.context_id, 'labels': list(session.labels),
        'candidate_roots': seed_identity(seed), 'root_sources': plain(root_sources),
        'collisions': collisions, 'comparison_conflicts': plain(result.comparison_conflicts),
        'proof_nodes': proofs, 'rules': [r.to_dict() for r in session.rules],
        'failed_state': complete_state_identity(result),
        'last_rule_presumed_wrong': False, 'confidence_based_repair': False,
        'failed_derived_facts_committed': False}
    return dict(body, conflict_id=digest(body))


def root_conflict_certificate(*, session, positive, negative, subset, strict_subset, root_sources, stage):
    """Contradictory candidate roots fail before constructing a valid Gamma.

    GammaSeed intentionally rejects overlapping signs. This certificate uses
    the two submitted root receipts directly; it does not forge a Chase run.
    """
    collisions, nodes = [], {}
    for row, label in np.argwhere(positive & negative):
        row, label = int(row), int(label)
        keys = ['%s:%d:%d' % (sign, row, label) for sign in ('pos', 'neg')]
        for key in keys:
            nodes[key] = {'origin': root_sources[key]['origin'], 'root_source': plain(root_sources[key]),
                'dependencies': [], 'rule_id': None, 'kind': key.split(':')[0], 'row': row, 'column': label}
        collisions.append({'document_index': row, 'label_index': label, 'label': session.labels[label],
            'positive_source': keys[0], 'negative_source': keys[1]})
    if not collisions:
        raise ValueError('direct root conflict certificate requires both signs of a fact')
    seed = SimpleNamespace(pos=positive, neg=negative, subset=subset, strict_subset=strict_subset)
    body = {'schema': 'paper36_outer_conflict_certificate_v1', 'outcome': 'Conflict',
        'stage': stage, 'Sigma': session.program_id, 'theta': session.snapshot_id,
        'context_id': session.context_id, 'labels': list(session.labels),
        'candidate_roots': seed_identity(seed), 'root_sources': plain(root_sources),
        'collisions': collisions, 'comparison_conflicts': [], 'proof_nodes': nodes,
        'rules': [r.to_dict() for r in session.rules], 'failed_state': None,
        'execution': 'direct_candidate_root_consistency_check_before_chase',
        'last_rule_presumed_wrong': False, 'confidence_based_repair': False,
        'failed_derived_facts_committed': False}
    return dict(body, conflict_id=digest(body))


def create_revision(parent_directory, *, reason, authority, rules, gamma, profile,
                    root_sources, parent_complete=None, context=None):
    """Bind an explicitly reviewed new (Sigma, Gamma, theta) to its parent.

    An inherited trusted root keeps its authenticated receipt. Every new or
    changed root requires a human/ground_truth receipt; derived, weak, and LLM
    facts cannot be silently promoted to validated Gamma.
    """
    parent = Path(parent_directory).resolve(strict=True)
    if not isinstance(reason, str) or not reason.strip() or not isinstance(authority, str) or not authority.strip():
        raise ValueError('revision requires an explicit reason and reviewing authority')
    complete_path = parent/'COMPLETE.json'
    if parent_complete is not None and bound_path(parent_complete) != complete_path:
        raise ValueError('parent_complete must authenticate this parent COMPLETE.json')
    complete = json.loads(complete_path.read_text())
    protocol_path = parent/'PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    if complete.get('protocol_id') != digest(protocol):
        raise ValueError('parent protocol is not bound by its completed run')
    corpus = protocol.get('corpus_identity')
    if (not isinstance(corpus, dict) or corpus.get('schema') != 'paper36_immutable_corpus_identity_v1'
            or corpus.get('sha256') != digest({key: value for key, value in corpus.items() if key != 'sha256'})):
        raise ValueError('parent lacks an authenticated complete corpus identity; automatic cross-run recovery is forbidden')
    if context is not None and corpus_identity(context) != corpus:
        raise ValueError('revision corpus documents/row identities/attributes differ from parent')
    if not isinstance(protocol.get('static_model_decisions_identity'), dict):
        raise ValueError('parent lacks an authenticated static model interpretation')
    static_models = (static_model_identity(context) if context is not None
        else protocol['static_model_decisions_identity'])
    profile.verify()
    if (list(profile.labels) != complete['final_profile']['labels']
            or profile.shape[0] != len(protocol['row_roles']) or gamma.shape != profile.shape):
        raise ValueError('revision must preserve the complete corpus and label axes')
    sources = plain(root_sources)
    if not isinstance(sources, dict) or set(sources) != root_keys(gamma):
        raise ValueError('revision sources must cover exactly the new Gamma roots')
    records = {'complete': file_record(complete_path), 'protocol': file_record(protocol_path)}
    inherited = {}
    if complete.get('root_sources') and complete.get('roots'):
        sources_path = bound_path(complete['root_sources'])
        roots_path = bound_path(complete['roots'])
        if sources_path != parent/'ROOT_SOURCES.json' or roots_path != parent/'ROOTS.npz':
            raise ValueError('parent root artifacts must belong to the parent run')
        records.update(root_sources=file_record(sources_path), roots=file_record(roots_path))
        inherited = json.loads(sources_path.read_text())
    reviewed = []
    for key, source in sources.items():
        if not isinstance(source, dict) or source.get('origin') not in ('human', 'ground_truth', 'validated_Gamma'):
            raise ValueError('revision Gamma requires trusted roots, never weak/derived/LLM testimony')
        if inherited.get(key) == source:
            continue
        if source['origin'] not in ('human', 'ground_truth') or not isinstance(source.get('receipt_id'), str) or not source['receipt_id'].strip():
            raise ValueError('new or revised roots require an explicit human/ground_truth receipt_id')
        reviewed.append(key)
    body = {'schema': 'paper36_explicit_outer_revision_v1', 'parent_directory': str(parent),
        'parent_artifacts': records, 'reason': reason, 'authority': authority,
        'Sigma': digest([r.to_dict() for r in rules]), 'Gamma': seed_identity(gamma),
        'profile_id': profile.profile_id, 'labels': list(profile.labels), 'shape': list(profile.shape),
        'root_sources': sources, 'reviewed_root_keys': sorted(reviewed),
        'corpus_identity': corpus,
        'static_model_decisions_identity': static_models,
        'row_roles': protocol['row_roles'], 'derived_labels_promoted_to_Gamma': False,
        'new_BO_selection_claimed': False, 'restart_from_explicit_roots': True}
    return dict(body, revision_id=digest(body))


def validate_revision(revision, *, rules, gamma, profile, root_sources, row_roles, context):
    revision = plain(revision)
    if not isinstance(revision, dict) or revision.get('schema') != 'paper36_explicit_outer_revision_v1':
        raise ValueError('explicit authenticated outer revision required')
    if revision.get('revision_id') != digest({k: v for k, v in revision.items() if k != 'revision_id'}):
        raise ValueError('revision binding changed')
    for record in revision['parent_artifacts'].values():
        bound_path(record)
    if revision.get('corpus_identity') != corpus_identity(context):
        raise ValueError('revision corpus documents/row identities/attributes differ from parent')
    if revision.get('static_model_decisions_identity') != static_model_identity(context):
        raise ValueError('static model decisions differ from the reviewed revision input')
    if (revision['Sigma'] != digest([r.to_dict() for r in rules]) or revision['Gamma'] != seed_identity(gamma)
            or revision['profile_id'] != profile.profile_id or revision['root_sources'] != plain(root_sources)
            or revision['labels'] != list(profile.labels) or revision['shape'] != list(profile.shape)
            or revision['row_roles'] != list(row_roles)):
        raise ValueError('revised rules/Gamma/profile/source/role inputs differ from reviewed binding')
    # Rebuild the policy check as well as checking a digest supplied by a caller.
    rebuilt = create_revision(revision['parent_directory'], reason=revision['reason'],
        authority=revision['authority'], rules=rules, gamma=gamma, profile=profile,
        root_sources=root_sources, parent_complete=revision['parent_artifacts']['complete'], context=context)
    if rebuilt != revision:
        raise ValueError('revision no longer matches its parent or root review policy')
    return copy.deepcopy(revision)


def validate_revision_budget(revision, budget):
    """A new input version cannot reset already consumed external-call quota."""
    parent = json.loads(bound_path(revision['parent_artifacts']['complete']).read_text())['query_budget']
    current = budget.receipt()
    previous_calls = parent['calls']
    if (current['limits'] != parent['limits'] or len(current['calls']) < len(previous_calls)
            or current['calls'][:len(previous_calls)] != previous_calls):
        raise ValueError('revision must preserve the parent query-budget limits and complete call history')
    return current


class RevisionChaseSession(TypedChaseSession):
    """Explicit reviewed Sigma provenance, without claiming a new BO run."""
    def __init__(self, *args, revision, **kwargs):
        if (revision.get('schema') != 'paper36_explicit_outer_revision_v1'
                or revision.get('revision_id') != digest({k: v for k, v in revision.items() if k != 'revision_id'})):
            raise ValueError('explicit revision binding changed before session construction')
        for record in revision['parent_artifacts'].values():
            bound_path(record)
        self._revision = copy.deepcopy(revision)
        super().__init__(*args, **kwargs)

    def _selection(self, directory):
        if directory is not None:
            return super()._selection(directory)
        if self.program_id != self._revision['Sigma']:
            raise ValueError('explicit revision does not authenticate this Sigma')
        return {'mode': 'explicit_author_revision', 'selection_complete_claim': False,
            'revision_id': self._revision['revision_id'], 'authority': self._revision['authority'],
            'reason': self._revision['reason'], 'new_BO_selection_claimed': False}
