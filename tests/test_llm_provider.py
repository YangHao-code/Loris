"""Offline provider contract tests: fake HTTP only, never real keys or APIs."""
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from loris.labeling.external import external_answer
from loris.labeling.llm_provider import _NoRedirect, main, parse_targets, propose

SYNTHETIC_KEY = 'synthetic-test-only-authorization-sentinel'


def request_value():
    return dict(row=7, row_id='doc:7', labels=['alpha', 'beta', 'gamma'],
                text='A synthetic document.', label_descriptions={'alpha': 'first topic'})


def response_value(targets=None, finish='stop'):
    return dict(model='served-test-model', usage={'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7},
                choices=[dict(finish_reason=finish, message=dict(content=json.dumps({'targets': [1, 0, 1] if targets is None else targets})))])


class FakeOpener:
    def __init__(self, result=None, error=None):
        self.result = response_value() if result is None else result
        self.error = error
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return io.StringIO(json.dumps(self.result))


def receipt_file(directory):
    return Path(directory) / '00000007.llm.api.json'


def output_text(directory):
    return '\n'.join(p.read_text() for p in Path(directory).iterdir() if p.is_file())


@pytest.mark.parametrize('targets', [[0, 0, 0], [1, 0, 1]])
def test_complete_zero_and_multilabel_proposals_reach_real_directory_callback(tmp_path, targets, capsys):
    opener = FakeOpener(response_value(targets))
    receipt = propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert receipt['status'] == 'proposal_ready_for_chase'
    assert receipt['network_attempts'] == 1 and receipt['retries'] == 0
    assert receipt['accepted_by_chase'] is False
    assert receipt['model_requested'] == 'deepseek-ai/DeepSeek-V4-Flash'
    assert receipt['model_returned'] == 'served-test-model'
    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == 'https://api.siliconflow.cn/v1/chat/completions'
    assert request.get_header('Authorization') == 'Bearer ' + SYNTHETIC_KEY
    payload = json.loads(request.data)
    assert payload['model'] == receipt['model_requested']
    assert payload['temperature'] == 0 and payload['response_format'] == {'type': 'json_object'}
    assert SYNTHETIC_KEY not in request.data.decode()
    seen = []
    def check(proposal):
        seen.append(proposal)
        return True
    answer, audit = external_answer(7, 'doc:7', request_value()['labels'], tmp_path, check)
    assert answer['targets'] == targets
    assert set(answer) == {'row_id', 'labels', 'targets', 'provider', 'provenance'}
    assert answer['provider'] == 'llm' and len(seen) == 1
    assert audit['llm_consistency_checks'] == audit['llm_accepted'] == 1
    assert audit['network_calls'] == 0  # File adapter itself does not send HTTP.
    assert SYNTHETIC_KEY not in output_text(tmp_path)
    captured = capsys.readouterr()
    assert SYNTHETIC_KEY not in captured.out + captured.err


def test_consistency_rejection_stays_pending_without_human_answer(tmp_path):
    propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=FakeOpener())
    seen = []
    def reject(proposal):
        seen.append(proposal)
        return False
    answer, receipt = external_answer(7, 'doc:7', request_value()['labels'], tmp_path, reject)
    assert answer is None and len(seen) == 1
    assert receipt['llm_consistency_checks'] == 1 and receipt['llm_accepted'] == 0
    assert receipt['status'] == 'pending_external_answer'


@pytest.mark.parametrize('positive,expected', [([], [0, 0, 0]), (['gamma', 'alpha'], [1, 0, 1])])
def test_positive_label_wire_format_is_canonical(tmp_path, positive, expected):
    result = response_value()
    result['choices'][0]['message']['content'] = json.dumps({'positive_labels': positive})
    receipt = propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY,
                      opener=FakeOpener(result))
    answer, audit = external_answer(7, 'doc:7', request_value()['labels'], tmp_path, lambda proposal: True)
    assert answer['targets'] == expected
    assert receipt['answer_format'] == 'positive_labels'
    assert receipt['thinking'] is False
    assert audit['llm_accepted'] == 1


def test_positive_label_wire_format_rejects_unknown_names(tmp_path):
    result = response_value()
    result['choices'][0]['message']['content'] = json.dumps({'positive_labels': ['unknown']})
    with pytest.raises(RuntimeError):
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=FakeOpener(result))
    assert not (tmp_path / '00000007.llm.json').exists()


@pytest.mark.parametrize('content', [
    '{"targets":[null,0,1]}', '{"targets":[true,0,1]}', '{"targets":[1.0,0,1]}',
    '{"targets":[1,0]}', '{"targets":[1,0,1,0]}', '{"targets":[-1,0,1]}',
    '{"targets":[2,0,1]}', '{"targets":[1,0,1],"explanation":"no"}',
    '[1,0,1]', '```json\n{"targets":[1,0,1]}\n```',
])
def test_invalid_complete_vectors_are_rejected(content):
    with pytest.raises(ValueError):
        parse_targets(content, 3)


@pytest.mark.parametrize('targets', [[None, 0, 1], [True, 0, 1], [1, 0]])
def test_invalid_api_vectors_do_not_create_answers(tmp_path, targets):
    opener = FakeOpener(response_value(targets))
    with pytest.raises(RuntimeError):
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert len(opener.calls) == 1
    assert not (tmp_path / '00000007.llm.json').exists()
    assert json.loads(receipt_file(tmp_path).read_text())['status'] == 'failed'


@pytest.mark.parametrize('finish', ['length', 'content_filter', 'tool_calls', None])
def test_nonstop_finish_is_rejected_even_when_vector_looks_valid(tmp_path, finish):
    opener = FakeOpener(response_value(finish=finish))
    with pytest.raises(RuntimeError):
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert len(opener.calls) == 1
    assert not (tmp_path / '00000007.llm.json').exists()


@pytest.mark.parametrize('key', [None, '', '   '])
def test_missing_key_fails_without_requests_or_receipts(tmp_path, key):
    opener = FakeOpener()
    with pytest.raises(ValueError):
        propose(request_value(), tmp_path / 'new', api_key=key, opener=opener)
    assert opener.calls == []
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('name', ['00000007.llm.json', '00000007.llm.api.json'])
def test_existing_answer_or_attempt_is_not_queried_again(tmp_path, name):
    (tmp_path / name).write_text('existing artifact unchanged')
    opener = FakeOpener()
    with pytest.raises(FileExistsError):
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert opener.calls == []
    assert (tmp_path / name).read_text() == 'existing artifact unchanged'


def test_dangling_existing_answer_path_is_not_queried_again(tmp_path):
    answer = tmp_path / '00000007.llm.json'
    answer.symlink_to(tmp_path / 'nonexistent_target')
    opener = FakeOpener()
    with pytest.raises(FileExistsError):
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert opener.calls == []
    assert answer.is_symlink()


@pytest.mark.parametrize('field', ['truth', 'ground_truth', 'targets', 'observed', 'gold'])
def test_request_rejects_truth_fields_before_network_or_output(tmp_path, field):
    value = request_value()
    value[field] = [1, 0, 1]
    opener = FakeOpener()
    with pytest.raises(ValueError):
        propose(value, tmp_path / 'new', api_key=SYNTHETIC_KEY, opener=opener)
    assert opener.calls == [] and not (tmp_path / 'new').exists()


def test_http_error_body_and_credentials_are_not_written_or_echoed(tmp_path, capsys):
    body = io.BytesIO(('vendor diagnostic ' + SYNTHETIC_KEY).encode())
    error = HTTPError('https://api.example/v1/chat/completions', 429,
                      'vendor echoed ' + SYNTHETIC_KEY, {}, body)
    opener = FakeOpener(error=error)
    with pytest.raises(RuntimeError) as caught:
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert len(opener.calls) == 1 and body.tell() == 0
    receipt = json.loads(receipt_file(tmp_path).read_text())
    assert receipt['status'] == 'provider_http_error' and receipt['http_status'] == 429
    assert receipt['network_attempts'] == 1 and receipt['retries'] == 0
    assert SYNTHETIC_KEY not in str(caught.value) + output_text(tmp_path)
    captured = capsys.readouterr()
    assert SYNTHETIC_KEY not in captured.out + captured.err


def test_connection_error_does_not_echo_transport_reason(tmp_path):
    opener = FakeOpener(error=URLError('diagnostic includes ' + SYNTHETIC_KEY))
    with pytest.raises(RuntimeError) as caught:
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert len(opener.calls) == 1
    assert SYNTHETIC_KEY not in str(caught.value) + output_text(tmp_path)


@pytest.mark.parametrize('field', ['model', 'usage'])
def test_untrusted_response_metadata_cannot_persist_credentials(tmp_path, field):
    result = response_value()
    result[field] = SYNTHETIC_KEY if field == 'model' else {'unexpected_diagnostic': SYNTHETIC_KEY, 'total_tokens': 7}
    opener = FakeOpener(result)
    try:
        propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    except RuntimeError:
        pass  # Rejecting malformed metadata is also acceptable.
    assert len(opener.calls) == 1
    assert SYNTHETIC_KEY not in output_text(tmp_path)


@pytest.mark.parametrize('url', [
    'http://api.example/v1', 'https://name:secret@api.example/v1',
    'https://api.example/v1?api_key=not-a-real-key', 'https://api.example/v1#fragment',
])
def test_unsafe_provider_urls_fail_before_request(tmp_path, url):
    opener = FakeOpener()
    with pytest.raises(ValueError):
        propose(request_value(), tmp_path / 'new', api_key=SYNTHETIC_KEY, base_url=url, opener=opener)
    assert not opener.calls and not (tmp_path / 'new').exists()


def test_redirect_handler_never_forwards_authorization():
    assert _NoRedirect().redirect_request(None, None, 302, 'redirect', {}, 'https://other.example') is None


def test_cli_missing_local_key_does_not_prompt_when_noninteractive(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('LORIS_TEST_PROVIDER_MISSING_KEY', raising=False)
    monkeypatch.setattr('loris.labeling.llm_provider.sys.stdin.isatty', lambda: False)
    value = tmp_path / 'request.json'
    value.write_text(json.dumps(request_value()))
    with pytest.raises(SystemExit) as caught:
        main(['--request', str(value), '--output-dir', str(tmp_path / 'out'),
              '--api-key-env', 'LORIS_TEST_PROVIDER_MISSING_KEY'])
    assert caught.value.code == 2
    assert not (tmp_path / 'out').exists()
    assert 'missing' in capsys.readouterr().err
