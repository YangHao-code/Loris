"""Credential-free offline checks for generation-bound identity and parse replay."""
import hashlib
import io
import json

import pytest

from loris.labeling import llm_provider as provider
from tests.test_llm_provider import FakeOpener, SYNTHETIC_KEY, request_value, response_value, output_text


@pytest.mark.parametrize('options', [
    {'model': 'different-model'}, {'base_url': 'https://other.example/v1'},
    {'max_tokens': 4096}, {'thinking': True}, {'thinking': None},
    {'answer_format': 'targets'}, {'system_prompt': 'A revised labeling instruction.'},
])
def test_generation_and_endpoint_changes_get_distinct_cache_keys(options):
    original = provider.proposal_cache_key(request_value())
    assert provider.proposal_cache_key(request_value(), **options) != original
    legacy = hashlib.sha256(json.dumps(request_value(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert original != legacy


def test_resolved_default_prompt_and_descriptions_are_bound(monkeypatch):
    original = provider.proposal_cache_key(request_value())
    monkeypatch.setattr(provider, 'POSITIVE_LABEL_SYSTEM', 'Changed default instructions.')
    assert provider.proposal_cache_key(request_value()) != original
    value = request_value()
    value['label_descriptions']['alpha'] = 'Different interpretation.'
    assert provider.proposal_cache_key(value) != provider.proposal_cache_key(request_value())


def test_identity_covers_exact_outgoing_payload_and_preserves_defaults(tmp_path):
    opener = FakeOpener()
    receipt = provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    wire = opener.calls[0][0].data
    identity = provider.proposal_cache_identity(request_value())
    assert json.loads(wire) == identity['payload']
    assert identity['payload']['messages'][0]['content'] == provider.POSITIVE_LABEL_SYSTEM
    assert identity['payload']['temperature'] == 0
    assert receipt['cache_identity_sha256'] == provider.proposal_cache_key(request_value())
    assert receipt['request_sha256'] == hashlib.sha256(wire).hexdigest()
    assert SYNTHETIC_KEY not in json.dumps(identity)


@pytest.mark.parametrize('content', [
    '{"positive_labels":["gamma","alpha"]}', '{"positive_labels":[]}', '{"targets":[1,0,1]}'
])
def test_saved_content_replays_the_exact_successful_vector(tmp_path, content):
    result = response_value()
    result['choices'][0]['message']['content'] = content
    receipt = provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=FakeOpener(result))
    artifact = tmp_path / receipt['response_artifact']['path']
    audit = json.loads(artifact.read_text())
    assert audit['message_content'] == content and not audit['content_redacted']
    assert audit['model_returned'] == result['model'] and audit['finish_reason'] == 'stop'
    assert audit['usage'] == result['usage']
    assert receipt['response_sha256'] == audit['response_sha256'] == hashlib.sha256(json.dumps(result).encode()).hexdigest()
    assert receipt['response_artifact']['sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    proposal = json.loads((tmp_path / '00000007.llm.json').read_text())
    assert provider.parse_targets(audit['message_content'], 3, request_value()['labels']) == proposal['targets']


@pytest.mark.parametrize('finish,content', [
    ('length', '{"positive_labels":["alpha"]}'), ('stop', '{"positive_labels":["unknown"]}'),
    ('stop', 'Not JSON'),
])
def test_failed_validation_still_keeps_identity_usage_and_content(tmp_path, finish, content):
    result = response_value(finish=finish)
    result['choices'][0]['message']['content'] = content
    with pytest.raises(RuntimeError):
        provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=FakeOpener(result))
    receipt = json.loads((tmp_path / '00000007.llm.api.json').read_text())
    audit = json.loads((tmp_path / receipt['response_artifact']['path']).read_text())
    assert receipt['status'] == 'failed'
    assert receipt['model_returned'] == audit['model_returned'] == result['model']
    assert audit['finish_reason'] == finish and audit['message_content'] == content
    assert receipt['usage'] == result['usage']
    assert not (tmp_path / '00000007.llm.json').exists()


def test_non_json_http_response_retains_only_digest(tmp_path):
    class InvalidJSON:
        def open(self, request, timeout):
            return io.BytesIO(('not JSON ' + SYNTHETIC_KEY).encode())
    with pytest.raises(RuntimeError):
        provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=InvalidJSON())
    receipt = json.loads((tmp_path / '00000007.llm.api.json').read_text())
    audit = json.loads((tmp_path / receipt['response_artifact']['path']).read_text())
    assert audit['response_sha256'] == hashlib.sha256(('not JSON ' + SYNTHETIC_KEY).encode()).hexdigest()
    assert audit['message_content'] is None
    assert SYNTHETIC_KEY not in output_text(tmp_path)


@pytest.mark.parametrize('field', ['content', 'model', 'finish'])
def test_untrusted_allowlisted_text_is_redacted_before_persistence(tmp_path, field):
    result = response_value()
    value = 'echoed ' + SYNTHETIC_KEY
    if field == 'content': result['choices'][0]['message']['content'] = value
    elif field == 'model': result['model'] = value
    else: result['choices'][0]['finish_reason'] = value
    with pytest.raises(RuntimeError):
        provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=FakeOpener(result))
    assert SYNTHETIC_KEY not in output_text(tmp_path)
    assert '[REDACTED_API_KEY]' in output_text(tmp_path)


def test_existing_response_audit_prevents_retry_or_overwrite(tmp_path):
    artifact = tmp_path / '00000007.llm.response.json'
    artifact.write_text('existing audit')
    opener = FakeOpener()
    with pytest.raises(FileExistsError):
        provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener)
    assert opener.calls == [] and artifact.read_text() == 'existing audit'


def test_fewshot_is_explicit_request_local_and_cache_bound(tmp_path):
    examples = [dict(row_id='train:1', document='A distinct training example.', positive_labels=['beta'])]
    options = dict(annotation_instructions='A dataset indexing profile.', annotation_examples=examples)
    before = provider.proposal_cache_identity(request_value())
    enhanced = provider.proposal_cache_identity(request_value(), **options)
    assert enhanced['payload']['messages'][0]['content'] == options['annotation_instructions']
    content = json.loads(enhanced['payload']['messages'][1]['content'])
    assert content['annotated_training_examples'] == [{'document': examples[0]['document'], 'positive_labels': ['beta']}]
    assert enhanced['annotation_examples'][0]['row_id'] == 'train:1'
    assert provider.proposal_cache_identity(request_value()) == before
    assert provider.proposal_cache_key(request_value(), **options) != provider.proposal_cache_key(request_value())
    changed = dict(options, annotation_examples=[dict(examples[0], positive_labels=[])])
    assert provider.proposal_cache_key(request_value(), **changed) != provider.proposal_cache_key(request_value(), **options)
    opener = FakeOpener()
    provider.propose(request_value(), tmp_path, api_key=SYNTHETIC_KEY, opener=opener, **options)
    assert json.loads(opener.calls[0][0].data) == enhanced['payload']


@pytest.mark.parametrize('bad', [
    {'document': 'Distinct text', 'positive_labels': ['unknown']},
    {'document': 'Distinct text', 'positive_labels': ['alpha', 'alpha']},
    {'document': 'Distinct text', 'positive_labels': ['alpha'], 'targets': [1, 0, 0]},
    {'document': request_value()['text'], 'positive_labels': ['alpha']},
    {'document': 'Distinct text', 'positive_labels': ['alpha'], 'row_id': request_value()['row_id']},
    {'document': 'Distinct text', 'positive_labels': ['alpha'], 'gold': True},
])
def test_query_truth_and_malformed_demonstrations_fail_before_network(tmp_path, bad):
    opener = FakeOpener()
    with pytest.raises(ValueError):
        provider.propose(request_value(), tmp_path/'new', api_key=SYNTHETIC_KEY,
                         opener=opener, annotation_examples=[bad])
    assert not opener.calls and not (tmp_path/'new').exists()
