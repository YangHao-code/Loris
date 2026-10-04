import json

import pytest

from loris._engine.rdl_reference import Doc, Label, Rule
from loris.labeling.cli import validate
from loris.labeling.runner import LabelingRun
from tests.test_batch_query_and_exports import configured
from tests.test_labeling import synthetic_same_trainer


@pytest.mark.parametrize('pool, expected_queries', [('uncovered', 0), ('all_unanswered', 3)])
def test_complete_model_coverage_does_not_end_annotation_rounds(tmp_path, pool, expected_queries):
    context, rules, manifest = configured(tmp_path)
    rules = rules + (Rule((Doc('x'),), Label('x', 'A')),)
    manifest['loop'].update(query_pool=pool, query_cadence='each_closure', label_budget=3)
    manifest['query']['strategy'] = 'uncertainty'
    for row in (1, 2, 3):
        (tmp_path / 'answers' / ('%08d.json' % row)).write_text(json.dumps(
            dict(row_id='r%d' % row, labels=['A', 'B'], targets=[1, 0])))
    batches = []

    def trainer(**arguments):
        batches.append(arguments['batch'].true_mask.copy())
        return synthetic_same_trainer(**arguments)

    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                     output=tmp_path / 'run', trainer=trainer)
    report = run.run()
    assert report['actual_label_requests'] == expected_queries
    assert report['query_policy']['pool'] == pool
    if pool == 'all_unanswered':
        selected = [event['rows'] for event in run.events if event['kind'] == 'query_batch_selected']
        assert selected == [[1, 2], [3]]
        assert len(batches) == 2
        assert batches[0][1:3].all() and batches[1][3].all()
        assert (run.gamma.pos | run.gamma.neg).all()
        assert run.gamma.pos[:, 0].all() and run.gamma.neg[:, 1].all()
    else:
        assert not batches


def test_budget_exhaustion_still_trains_last_batch(tmp_path):
    context, rules, manifest = configured(tmp_path)
    rules = rules + (Rule((Doc('x'),), Label('x', 'A')),)
    manifest['loop'].update(query_pool='all_unanswered', query_cadence='each_closure', label_budget=2)
    manifest['query']['strategy'] = 'uncertainty'
    for row in (1, 2, 3):
        (tmp_path / 'answers' / ('%08d.json' % row)).write_text(json.dumps(
            dict(row_id='r%d' % row, labels=['A', 'B'], targets=[1, 0])))
    run = LabelingRun(context=context, rules=rules, manifest=manifest,
                     output=tmp_path / 'run', trainer=synthetic_same_trainer)
    report = run.run()
    assert report['status'] == 'label_budget_exhausted'
    assert report['actual_label_requests'] == 2
    assert report['training_attempts'] == 1
    assert not (run.gamma.pos[3] | run.gamma.neg[3]).any()


def test_invalid_query_pool_fails_validation(tmp_path):
    context, rules, manifest = configured(tmp_path)
    manifest['loop']['query_pool'] = 'all_truth'
    with pytest.raises(ValueError, match='query_pool'):
        validate(manifest)
