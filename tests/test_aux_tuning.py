import numpy as np
import pytest
import json

from loris.experiments.aux_tuning import build_batch, load_assets, planned_arms, pseudo_supervision, scores
from loris.discovery.manifest import file_record


def test_pseudo_labels_never_overwrite_known_or_make_unknown_negative():
    probabilities = np.array([[.99, .5, .01], [.8, .98, .2]])
    known = np.array([[True, False, False], [False, False, True]])
    targets, observed, weights, audit = pseudo_supervision(probabilities, known, method='self_train')
    assert not observed[known].any()
    assert not observed[0, 1] and weights[0, 1] == 0
    assert observed[0, 2] and targets[0, 2] == 0
    assert observed[1, 1] and targets[1, 1] == 1
    assert audit['weak_positive'] == audit['weak_negative'] == 1


def test_adaptive_thresholds_are_bounded_and_missing_classes_stay_conservative():
    probabilities = np.array([[.99, .99, .6], [.98, .84, .6], [.97, .83, .6]])
    known = np.zeros_like(probabilities, bool)
    _, fixed, _, _ = pseudo_supervision(probabilities, known, method='self_train')
    _, adaptive, _, audit = pseudo_supervision(probabilities, known, method='adaptive_self_train')
    assert np.asarray(audit['thresholds_positive']).min() >= .8
    assert audit['thresholds_positive'][2] == .95
    assert adaptive[1, 1] and not fixed[1, 1]
    assert not adaptive[:, 2].any()


def test_supervised_control_has_no_pseudo_observations():
    probabilities = np.array([[.99, .01], [.01, .99]])
    known = np.array([[True, True], [False, False]])
    targets = np.array([[0, 1], [0, 0]])
    batch = build_batch(probabilities, targets, known, dict(method='supervised', initialization='m0'))
    assert np.array_equal(batch.observed, known)
    assert np.array_equal(batch.targets[known], targets[known])
    assert (batch.weights[known] == 1).all()
    assert not batch.weak_mask.any()


def test_incremental_control_retains_fresh_answers_and_only_changed_weak_targets():
    probabilities = np.array([[.99, .01], [.99, .01], [.99, .01]])
    known = np.array([[True, True], [False, False], [False, False]])
    targets = np.array([[0, 1], [0, 0], [0, 0]])
    configuration = dict(method='self_train', initialization='previous_delta')
    first = build_batch(probabilities, targets, known, configuration, fresh_rows=[0])
    known[1] = True
    targets[1] = [1, 1]
    second = build_batch(probabilities, targets, known, configuration, fresh_rows=[1], previous=first.feedback_state)
    assert second.true_mask[1].all()
    assert not second.weak_mask[2].any()
    assert np.array_equal(second.targets[1], targets[1])


@pytest.mark.parametrize('bad', [float('nan'), -.1, 1.1])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        pseudo_supervision(np.array([[bad]]), np.array([[False]]), method='self_train')


def test_matrix_separates_one_update_epochs_and_multiple_update_initialization():
    arms = planned_arms()
    assert len({arm['id'] for arm in arms}) == len(arms)
    assert {arm['epochs'] for arm in arms if arm['method'] == 'supervised'} == {1, 3, 5}
    assert {arm['initialization'] for arm in arms if arm['rounds'] == 3} == {'m0', 'previous', 'previous_delta'}
    score = scores([[1, 0], [0, 1]], [[.9, .1], [.1, .9]])
    assert score['mean_f1'] == 1


def test_official_test_selection_requires_explicit_role_and_train_only_queries(tmp_path):
    texts = tmp_path / 'texts.jsonl'
    texts.write_text('\n'.join(json.dumps(dict(row_id=row, text=row))
        for row in ['example:train:0', 'example:test:0']))
    np.savez(tmp_path / 'training.npz', row_ids=['example:train:0'], labels=['label'],
        targets=[[0]], observed=[[False]])
    np.savez(tmp_path / 'tuning.npz', row_ids=['example:test:0'], labels=['label'], targets=[[1]])
    np.savez(tmp_path / 'queries.npz', row_ids=['example:train:0'], labels=['label'], targets=[[1]])
    manifest = dict(schema='loris_train_only_aux_tuning_v1', labels=['label'], audit_row_ids=[],
        rule_q_row_ids=[], files={name: file_record(tmp_path / filename) for name, filename in
            [('texts', 'texts.jsonl'), ('training', 'training.npz'), ('tuning', 'tuning.npz'), ('queries', 'queries.npz')]})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='split roles'):
        load_assets(path)
    manifest['selection_role'] = 'official_test_selection'
    path.write_text(json.dumps(manifest))
    assert load_assets(path)[0]['selection_role'] == 'official_test_selection'
    np.savez(tmp_path / 'queries.npz', row_ids=['example:test:0'], labels=['label'], targets=[[1]])
    manifest['files']['queries'] = file_record(tmp_path / 'queries.npz')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='confined to the training pool'):
        load_assets(path)
