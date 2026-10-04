import json

import numpy as np
import pytest

from loris.labeling.server_broker import full_answer, process_request


def fixture(tmp_path):
    suite = tmp_path / 'suite'
    for name in ['answers', 'requests', 'ready']:
        (suite / 'external' / name).mkdir(parents=True)
    labels = ['one', 'two']
    (suite / 'TARGET_AXIS.json').write_text(json.dumps(dict(row_ids=['target:0'], rows=[10], labels=labels)))
    request = dict(row=10, row_id='target:0', labels=labels, text='document')
    truth = tmp_path / 'truth.npz'
    np.savez(truth, row_ids=['target:0', 'target:1'], labels=labels,
        targets=[[1, 0], [0, 1]], observed=np.ones((2, 2), bool))
    return suite, request, truth


def test_server_human_discloses_requested_full_vector_only(tmp_path):
    suite, request, truth = fixture(tmp_path)
    value = full_answer(request, truth)
    assert value['targets'] == [1, 0]
    assert value['row_id'] == 'target:0'
    request['row_id'] = 'missing'
    with pytest.raises(ValueError):
        full_answer(request, truth)


def test_server_fallback_requires_llm_receipt_and_publishes_complete_answer(tmp_path):
    suite, request, truth = fixture(tmp_path)
    path = suite / 'external/requests/00000010.human.json'
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match='preceding LLM'):
        process_request(suite, path, truth, 'unused-test-key', tmp_path / 'cache')
    (suite / 'external/answers/00000010.llm.api.json').write_text(json.dumps(dict(status='failed')))
    process_request(suite, path, truth, 'unused-test-key', tmp_path / 'cache')
    answer = json.loads((suite / 'external/answers/00000010.human.json').read_text())
    assert answer['targets'] == [1, 0]
    assert (suite / 'external/ready/00000010.human.json').is_file()


def test_existing_identical_llm_response_is_reused_without_network(tmp_path):
    suite, request, truth = fixture(tmp_path)
    path = suite / 'external/requests/00000010.llm.json'
    path.write_text(json.dumps(request))
    answer = dict(row_id=request['row_id'], labels=request['labels'], targets=[1, 0], provider='llm')
    (suite / 'external/answers/00000010.llm.json').write_text(json.dumps(answer))
    (suite / 'external/answers/00000010.llm.api.json').write_text(json.dumps(dict(status='proposal_ready_for_chase')))
    process_request(suite, path, truth, 'unused-test-key', tmp_path / 'cache', sources=[suite])
    assert json.loads((suite / 'external/answers/00000010.llm.json').read_text()) == answer
    assert (suite / 'external/ready/00000010.llm.json').is_file()
