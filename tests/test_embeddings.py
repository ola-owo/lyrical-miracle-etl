"""Tests for data_loader.embeddings helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest
from google.genai.errors import ClientError
from google.genai.types import JobState

from data_loader import embeddings
from tests.utils import unwrap

retrieve_batch_file_job = unwrap(embeddings.retrieve_batch_file_job)
refresh_batch_jobs = unwrap(embeddings.refresh_batch_jobs)

BATCH_FILE = b'{"key": "song-1", "response": {"embedding": {"values": [0.5, 0.25]}}}\n'


def _batch_job(state, name='job-1', dest=None, error=None):
    return SimpleNamespace(
        name=name,
        display_name='test job',
        state=state,
        dest=dest,
        completion_stats=None,
        error=error,
        output_info=None,
        src=None,
        model='gemini-embedding-2',
    )


def _succeeded_job(name='job-1', file_name='jobs/job-1.jsonl'):
    dest = SimpleNamespace(file_name=file_name, inlined_embed_content_responses=None)
    return _batch_job(JobState.JOB_STATE_SUCCEEDED, name=name, dest=dest)


def _mock_client(job):
    client = MagicMock()
    client.batches.get.return_value = job
    return client


def test_check_batch_inputs_accepts_valid_batch():
    texts = pl.DataFrame({'content': ['a', 'b']})
    assert embeddings._check_batch_inputs(texts) is None


def test_check_batch_inputs_rejects_oversized_batch():
    texts = pl.DataFrame({'content': ['a'] * (embeddings.MAX_JOB_SIZE + 1)})
    with pytest.raises(ValueError, match='too large'):
        embeddings._check_batch_inputs(texts)


def test_check_batch_inputs_rejects_nulls():
    texts = pl.DataFrame({'content': ['a', None]})
    with pytest.raises(ValueError, match='Null batch inputs'):
        embeddings._check_batch_inputs(texts)


def test_build_prompt_plain_content():
    res = pl.select(embeddings._build_prompt(pl.lit('hello')))
    assert res.item() == 'title: none | text: hello'


def test_build_prompt_with_title():
    res = pl.select(embeddings._build_prompt(pl.lit('hello'), pl.lit('Song')))
    assert res.item() == 'title: Song | text: hello'


def test_build_prompt_strips_pipe_chars():
    res = pl.select(embeddings._build_prompt(pl.lit('he|llo'), pl.lit('So|ng')))
    assert res.item() == 'title: Song | text: hello'


def test_build_prompt_with_task():
    res = pl.select(
        embeddings._build_prompt(
            pl.lit('hello'), task=embeddings.EmbeddingTask.CLUSTERING
        )
    )
    assert res.item() == 'task: clustering | query: hello'


def test_build_prompt_rejects_invalid_task():
    with pytest.raises(ValueError):
        embeddings._build_prompt(pl.lit('hello'), task='bogus')


def test_get_job_state_returns_state():
    job = SimpleNamespace(state=JobState.JOB_STATE_SUCCEEDED)
    assert embeddings._get_job_state(job) == JobState.JOB_STATE_SUCCEEDED


def test_get_job_state_requires_state():
    with pytest.raises(ValueError, match='Job state missing'):
        embeddings._get_job_state(SimpleNamespace(state=None))


def test_job_to_dict_drops_heavy_fields():
    job = _batch_job(JobState.JOB_STATE_SUCCEEDED, dest='the destination')

    res = embeddings._job_to_dict(job)

    assert res == {
        'name': 'job-1',
        'display_name': 'test job',
        'state': JobState.JOB_STATE_SUCCEEDED,
        'model': 'gemini-embedding-2',
    }


def test_job_to_dict_tolerates_missing_fields():
    assert embeddings._job_to_dict(SimpleNamespace(name='job-1')) == {'name': 'job-1'}


def test_extract_from_batch_file_parses_embeddings():
    client = _mock_client(_succeeded_job())
    client.files.download.return_value = BATCH_FILE

    res = embeddings._extract_from_batch_file('job-1', client=client)

    assert res['id'].to_list() == ['song-1']
    assert res['embedding'].to_list() == [[0.5, 0.25]]
    client.batches.delete.assert_not_called()


def test_extract_from_batch_file_deletes_finished_job():
    client = _mock_client(_succeeded_job())
    client.files.download.return_value = BATCH_FILE

    embeddings._extract_from_batch_file('job-1', client=client, delete_finished=True)

    client.batches.delete.assert_called_once_with(name='job-1')


def test_extract_from_batch_file_keeps_unfinished_job():
    job = _batch_job(JobState.JOB_STATE_RUNNING)
    client = _mock_client(job)

    res = embeddings._extract_from_batch_file(
        'job-1', client=client, delete_finished=True
    )

    assert res is None
    client.batches.delete.assert_not_called()


def test_extract_from_batch_file_skips_job_without_destination():
    client = _mock_client(_batch_job(JobState.JOB_STATE_SUCCEEDED))

    assert embeddings._extract_from_batch_file('job-1', client=client) is None


def test_extract_from_batch_file_skips_inline_job():
    dest = SimpleNamespace(
        file_name=None, inlined_embed_content_responses=[{'key': 'song-1'}]
    )
    client = _mock_client(_batch_job(JobState.JOB_STATE_SUCCEEDED, dest=dest))

    assert embeddings._extract_from_batch_file('job-1', client=client) is None


def test_extract_from_batch_file_skips_job_without_file():
    dest = SimpleNamespace(file_name=None, inlined_embed_content_responses=None)
    client = _mock_client(_batch_job(JobState.JOB_STATE_SUCCEEDED, dest=dest))

    assert embeddings._extract_from_batch_file('job-1', client=client) is None


def test_retrieve_batch_file_job_yields_finished_jobs(monkeypatch):
    extracted = pl.DataFrame({'id': [1], 'embedding': [[0.1, 0.2]]})

    def fake_extract(job_name, client=None, delete_finished=False):
        return extracted if job_name == 'done' else None

    monkeypatch.setattr(embeddings, '_extract_from_batch_file', fake_extract)
    jobs = [
        {'name': 'done', 'state': 'JOB_STATE_SUCCEEDED'},
        {'name': 'pending', 'state': 'JOB_STATE_RUNNING'},
    ]

    res = list(retrieve_batch_file_job(jobs, client=MagicMock()))

    assert res[0].to_dicts() == [{'id': 1, 'embedding': [0.1, 0.2]}]
    assert len(res) == 1


def test_retrieve_batch_file_job_passes_delete_flag(monkeypatch):
    seen = {}

    def fake_extract(job_name, client=None, delete_finished=False):
        seen['delete_finished'] = delete_finished

    monkeypatch.setattr(embeddings, '_extract_from_batch_file', fake_extract)

    list(
        retrieve_batch_file_job(
            [{'name': 'job-1', 'state': 'JOB_STATE_RUNNING'}],
            client=MagicMock(),
            delete_finished=False,
        )
    )

    assert seen['delete_finished'] is False


def test_refresh_batch_jobs_yields_refreshed_records():
    client = MagicMock()
    client.batches.get.return_value = SimpleNamespace(
        name='job-1',
        display_name='d',
        state='JOB_STATE_RUNNING',
        dest='the destination',
        model='m',
    )

    res = list(refresh_batch_jobs([{'name': 'job-1'}], client=client))

    assert res == [
        {
            'name': 'job-1',
            'display_name': 'd',
            'state': 'JOB_STATE_RUNNING',
            'model': 'm',
        }
    ]
    client.batches.get.assert_called_once_with(name='job-1')


def test_refresh_batch_jobs_skips_client_errors():
    client = MagicMock()
    client.batches.get.side_effect = ClientError(
        400, {'error': {'message': 'bad', 'status': 'INVALID_ARGUMENT'}}
    )

    assert list(refresh_batch_jobs([{'name': 'deleted-job'}], client=client)) == []


def test_make_client_uses_explicit_key(monkeypatch):
    created = {}

    def fake_client(api_key=None, **kwargs):
        created['api_key'] = api_key
        return created

    monkeypatch.setattr(embeddings.genai, 'Client', fake_client)

    assert embeddings._make_client('explicit-key') == created
    assert created['api_key'] == 'explicit-key'


def test_make_client_falls_back_to_secret(monkeypatch):
    monkeypatch.setenv('SOURCES__GEMINI__API_KEY', 'secret-key')
    monkeypatch.setattr(
        embeddings.genai,
        'Client',
        lambda api_key=None, **kwargs: SimpleNamespace(api_key=api_key),
    )

    assert embeddings._make_client().api_key == 'secret-key'
