"""Materialize the complete declared raw pattern domain before F1 screening."""
from dataclasses import replace
import numpy as np

from loris.predicates import FreqPredicate
from loris._engine.rlearner_config import PatternSpec, ParameterDomain
from .patterns import PatternCandidate, FitLabelAssigner


def pattern_instances(templates):
    """All declared frequency values/operators, with no coverage/entropy gate."""
    for template in templates:
        if not isinstance(template, PatternSpec):
            raise TypeError('raw PatternSpec templates required')
        if not isinstance(template.predicate, FreqPredicate):
            yield template
            continue
        domain = template.frequency_domain
        if domain.values:
            values = tuple(domain.values)
        elif domain.integer:
            values = range(int(domain.low), int(domain.high) + 1)
        elif domain.low == domain.high:
            values = (domain.low,)
        else:
            raise ValueError('frequency instance screening requires an explicit finite domain')
        for op in template.frequency_ops:
            for eta in values:
                yield PatternSpec(template.key + '::' + op + '::' + str(eta),
                    replace(template.predicate, op=op, eta=eta),
                    frequency_domain=ParameterDomain(values=(eta,), integer=domain.integer),
                    frequency_ops=(op,))


def exact_pattern_mask(predicate, docs, attribute_presence=None):
    """Re-evaluate the actual predicate with the Chase missing-attribute gate.

    An absent attribute supplies no text evidence, including for Freq == 0.
    An explicit presence interpretation takes precedence over empty strings.
    No old pattern mask or selected-candidate cache is consulted.
    """
    docs = tuple(docs)
    attr = predicate.attr
    presence = np.asarray((attribute_presence or {}).get(attr,
        [bool(doc.get_attr(attr)) for doc in docs]))
    if presence.shape != (len(docs),) or not np.isin(presence, [0, 1]).all():
        raise ValueError('complete binary pattern attribute presence required')
    result = np.fromiter((bool(predicate(d)) for d in docs), bool, len(docs))
    return result & presence.astype(bool)


def build_pattern_candidates(templates, *, fit_docs, control_docs, fit_truth,
                             fit_row_ids, fit_baseline_predictions, input_provenance,
                             fit_attribute_presence=None, control_attribute_presence=None):
    """Fit candidate label associations only on fit; compute both full masks.

    A caller supplies the original extractor vocabulary, not an old selected
    P_i. Raw templates, their extraction provenance and order remain scientific
    inputs. Exact predicate implementations are shared with the Chase engine.
    """
    if not isinstance(input_provenance, dict) or input_provenance.get('raw_unfiltered_vocabulary') is not True:
        raise ValueError('declare original unfiltered extractor vocabulary provenance')
    fit_docs, control_docs = tuple(fit_docs), tuple(control_docs)
    if np.asarray(fit_truth).shape[0] != len(fit_docs):
        raise ValueError('fit label rows differ from fit documents')
    assign = FitLabelAssigner(fit_truth,row_ids=fit_row_ids,
                             baseline_predictions=fit_baseline_predictions)
    for spec in pattern_instances(templates):
        fit_mask = exact_pattern_mask(spec.predicate, fit_docs, fit_attribute_presence)
        control_mask = exact_pattern_mask(spec.predicate, control_docs, control_attribute_presence)
        labels, provenance = assign(fit_mask)
        provenance = dict(provenance, raw_extraction=input_provenance,
            frequency_instance_exact=True, legacy_coverage_entropy_prefilter=False,
            missing_attribute_semantics='false, matching Chase primitive masks')
        yield PatternCandidate(spec.key, spec.to_dict(), control_mask, labels, provenance)
