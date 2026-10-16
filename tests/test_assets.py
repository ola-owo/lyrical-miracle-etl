"""Asset-level tests: assets run through `dg.materialize` with externals patched."""

from unittest.mock import MagicMock

import dagster as dg
import numpy as np
import pendulum as pn
import polars as pl
import pytest

from data_loader import big5, clustering, embeddings, genius, lastfm
from data_loader.definitions import N_EMBED_JOBS, defs
from tests.utils import FakeResource

EMBED_CASES = [
    pytest.param(
        None, 'lyrics_embed', 'lyrics_embed_jobs', 'lyrics_no_embed', id='plain'
    ),
    pytest.param(
        embeddings.EmbeddingTask.CLUSTERING,
        'lyrics_embed_clustering',
        'lyrics_embed_clustering_jobs',
        'lyrics_no_embed_clustering',
        id='clustering',
    ),
]

CLUSTER_EMBEDDINGS = pl.DataFrame(
    {
        'id': [1, 2, 3, 4],
        'embedding': [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
    },
    schema={'id': pl.Int64, 'embedding': pl.Array(pl.Float64, 2)},
)


def _asset(key: str) -> dg.AssetsDefinition:
    return defs.get_assets_def(dg.AssetKey(key.split('/')))


def _row_counts(result: dg.ExecuteInProcessResult):
    events = result.get_asset_materialization_events()
    assert len(events) == 1
    return events[0].event_specific_data.materialization.metadata['row_counts'].data


def test_scrobbles_asset_uses_partition_window(monkeypatch, fake_pipelines):
    fake_pipelines.row_counts = {'scrobbles': 3}
    calls = {}

    def fake_scrobbles(start=None, end=None, limit=1000):
        calls['start'], calls['end'] = start, end
        return FakeResource('scrobbles')

    monkeypatch.setattr(lastfm, 'scrobbles', fake_scrobbles)

    res = dg.materialize([lastfm.scrobbles_asset], partition_key='2026-01-25')

    assert res.success
    assert calls['start'] == pn.datetime(2026, 1, 25, tz='UTC')
    assert calls['end'] == pn.datetime(2026, 2, 1, tz='UTC')
    assert _row_counts(res) == {'scrobbles': 3}

    (created,) = fake_pipelines
    assert created.args == ('scrobbles',)
    assert created.kwargs['dataset_name'] == 'lastfm'

    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': 'scrobbles',
        'write_disposition': 'merge',
        'primary_key': ('url', 'dt'),
        'loader_file_format': 'parquet',
    }
    assert run.source.step_names() == ['filter', 'map', 'map']


def test_matches_from_dataset_asset_loads_matches(fake_sql_table, fake_pipelines):
    fake_pipelines.row_counts = {'genius_matches': 2}

    res = dg.materialize([_asset('lastfm/genius_matches_from_dataset')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs == {
        'table': 'songs_not_searched',
        'schema': 'lastfm',
        'backend': 'pyarrow',
    }
    assert source.appended == [genius.match_to_dataset]

    (created,) = fake_pipelines
    assert created.args == ('genius_search',)
    (run,) = created.pipeline.runs
    assert run.source is source
    assert run.kwargs == {
        'dataset_name': 'lastfm',
        'table_name': 'genius_matches',
        'write_disposition': 'merge',
        'primary_key': ('song', 'artist'),
        'loader_file_format': 'parquet',
    }
    assert _row_counts(res) == {'genius_matches': 2}


def test_genius_searches_asset_appends_search_results(fake_sql_table, fake_pipelines):
    res = dg.materialize([_asset('lastfm/genius_searches')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == 'songs_not_searched'
    assert source.appended == [genius.genius_search]

    (created,) = fake_pipelines
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'dataset_name': 'lastfm',
        'table_name': 'genius_searches',
        'write_disposition': 'append',
        'loader_file_format': 'parquet',
    }


def test_matches_from_search_asset_merges_matches(fake_sql_table, fake_pipelines):
    res = dg.materialize([_asset('lastfm/genius_matches_from_search')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == 'searches_not_matched'
    assert source.appended == [genius.match_search_results]

    (created,) = fake_pipelines
    assert created.kwargs['dataset_name'] == 'lastfm'
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': 'genius_matches',
        'write_disposition': 'merge',
        'primary_key': ('song', 'artist'),
        'loader_file_format': 'parquet',
    }


def test_songs_asset_merges_new_metadata(fake_sql_table, fake_pipelines):
    res = dg.materialize([_asset('genius/songs')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == 'genius_matches_without_metadata'
    assert source.call.kwargs['schema'] == 'lastfm'
    assert source.appended == [genius.get_song_metadata]

    (created,) = fake_pipelines
    assert created.args == ('genius_song_meta',)
    assert created.kwargs['dataset_name'] == 'genius'
    assert created.kwargs['import_schema_path'] == 'schemas/import'
    assert created.kwargs['export_schema_path'] == 'schemas/export'

    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': 'songs',
        'write_disposition': 'merge',
        'primary_key': 'id',
        'loader_file_format': 'parquet',
    }


def test_lyrics_asset_loads_new_lyrics(fake_sql_table, fake_pipelines):
    res = dg.materialize([_asset('genius/lyrics')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == 'songs_no_lyrics'
    assert source.call.kwargs['schema'] == 'genius'
    assert source.appended == [genius.get_lyrics]

    (created,) = fake_pipelines
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': 'lyrics',
        'loader_file_format': 'parquet',
        'primary_key': 'id',
    }


def test_big5_ping_asset_wakes_endpoint(monkeypatch):
    aiplatform = MagicMock()
    monkeypatch.setattr(big5, 'aiplatform', aiplatform)

    res = dg.materialize([big5.big5_ping_asset])

    assert res.success
    aiplatform.init.assert_called_once_with(
        project='test-project', location='test-region'
    )
    aiplatform.Endpoint.assert_called_once_with(
        'projects/test-project/locations/test-region/endpoints/test-big5-endpoint'
    )
    aiplatform.Endpoint.return_value.predict.assert_called_once_with(instances=[])


def test_lyrics_big5_asset_merges_predictions(
    monkeypatch, fake_sql_table, fake_pipelines
):
    predict = {}

    def fake_predict_online():
        predict['resource'] = FakeResource('big5_predict_online')
        return predict['resource']

    monkeypatch.setattr(big5, 'big5_predict_online', fake_predict_online)

    res = dg.materialize([_asset('genius/lyrics_big5')])

    assert res.success
    (source,) = fake_sql_table
    assert source.call.kwargs == {
        'table': 'lyrics_no_big5',
        'schema': 'genius',
        'included_columns': ['id', 'lyrics'],
        'backend': 'pyarrow',
    }
    assert source.step_names() == ['map']
    assert source.appended == [predict['resource']]

    (created,) = fake_pipelines
    assert created.args == ('big5',)
    assert created.kwargs['dataset_name'] == 'genius'
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': 'lyrics_big5',
        'primary_key': 'id',
        'write_disposition': 'merge',
        'loader_file_format': 'parquet',
    }


def _patch_clustering(monkeypatch, embeddings_df):
    calls: dict = {}

    def fake_get_embeddings(fetch_all=False):
        calls['fetch_all'] = fetch_all
        return embeddings_df

    def fake_write(labels, centroids, replace=True):
        calls['labels'] = labels
        calls['centroids'] = centroids
        calls['replace'] = replace
        return {'lyrics_clusters': labels.height}

    monkeypatch.setattr(clustering, 'get_embeddings', fake_get_embeddings)
    monkeypatch.setattr(clustering, 'write_clustering', fake_write)
    return calls


def test_lyrics_clusters_asset_full_recluster(monkeypatch):
    calls = _patch_clustering(monkeypatch, CLUSTER_EMBEDDINGS)

    res = dg.materialize(
        [clustering.lyrics_clusters_asset],
        run_config=dg.RunConfig(
            ops={
                'genius__lyrics_clusters': {
                    'config': {'replace': True, 'n_clusters': 2}
                }
            }
        ),
    )

    assert res.success
    assert calls['fetch_all'] is True
    assert calls['replace'] is True
    assert calls['labels']['id'].to_list() == [1, 2, 3, 4]
    assert len(set(calls['labels']['cluster'].to_list())) == 2
    assert calls['centroids'].height == 2
    assert calls['centroids']['centroid'].list.len().to_list() == [2, 2]


def test_lyrics_clusters_asset_incremental_uses_existing_centroids(monkeypatch):
    calls = _patch_clustering(monkeypatch, CLUSTER_EMBEDDINGS)
    monkeypatch.setattr(
        clustering,
        '_get_centroids',
        lambda: np.array([[0.0, 0.0], [10.0, 10.0]]),
    )

    res = dg.materialize([clustering.lyrics_clusters_asset])

    assert res.success
    assert calls['fetch_all'] is False
    assert calls['replace'] is False
    assert calls['labels']['cluster'].to_list() == [0, 0, 1, 1]


def test_lyrics_clusters_asset_skips_empty_embeddings(monkeypatch):
    empty = pl.DataFrame(
        schema={
            'id': pl.Int64,
            'embedding': pl.Array(pl.Float64, clustering.EMBEDDING_DIM),
        }
    )
    calls = _patch_clustering(monkeypatch, empty)

    res = dg.materialize([clustering.lyrics_clusters_asset])

    assert res.success
    assert 'labels' not in calls
    assert _row_counts(res) == {}


@pytest.mark.parametrize('task, embed_table, jobs_table, lyrics_table', EMBED_CASES)
def test_embed_refresh_asset_replaces_unfinished_jobs(
    task,
    embed_table,
    jobs_table,
    lyrics_table,
    monkeypatch,
    fake_sql_table,
    fake_pipelines,
):
    fake_pipelines.row_counts = {jobs_table: 2}
    client = MagicMock()
    monkeypatch.setattr(embeddings, '_make_client', lambda: client)
    calls = {}

    def fake_refresh_batch_jobs(jobs, client=None):
        # dlt binds the piped job records to `jobs` at runtime; here the
        # asset's call-time argument (the Gemini client) lands there instead
        calls['first_arg'] = jobs
        return FakeResource('refresh_batch_jobs')

    monkeypatch.setattr(embeddings, 'refresh_batch_jobs', fake_refresh_batch_jobs)

    res = dg.materialize([_asset(f'genius/{embed_table}_jobs_refresh')])

    assert res.success
    assert calls['first_arg'] is client
    assert _row_counts(res) == {jobs_table: 2}
    (source,) = fake_sql_table
    assert source.call.kwargs == {
        'table': jobs_table,
        'schema': 'genius',
        'included_columns': ['name', 'state'],
        'backend': 'sqlalchemy',
    }
    (filter_fn,) = source.filters()
    assert filter_fn({'name': 'a', 'state': 'JOB_STATE_RUNNING'})
    assert not filter_fn({'name': 'b', 'state': 'JOB_STATE_SUCCEEDED'})

    (created,) = fake_pipelines
    suffix = '' if task is None else f'_{task.replace(" ", "_")}'
    assert created.args == (f'embed_lyrics{suffix}',)
    assert created.kwargs['dataset_name'] == 'genius'
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': jobs_table,
        'write_disposition': 'replace',
        'primary_key': 'name',
    }


@pytest.mark.parametrize('task, embed_table, jobs_table, lyrics_table', EMBED_CASES)
def test_embed_extract_asset_loads_finished_jobs(
    task,
    embed_table,
    jobs_table,
    lyrics_table,
    monkeypatch,
    fake_sql_table,
    fake_pipelines,
):
    client = MagicMock()
    monkeypatch.setattr(embeddings, '_make_client', lambda: client)
    calls = {}

    def fake_retrieve_batch_file_job(jobs, client=None, delete_finished=True):
        calls['first_arg'] = jobs
        calls['delete_finished'] = delete_finished
        return FakeResource('retrieve_batch_file_job')

    monkeypatch.setattr(
        embeddings, 'retrieve_batch_file_job', fake_retrieve_batch_file_job
    )

    loaded = []

    def fake_load_all_normalized(pipeline, tables, clear_after_load=True):
        loaded.append((pipeline, tables, clear_after_load))

    monkeypatch.setattr(embeddings, '_load_all_normalized', fake_load_all_normalized)

    res = dg.materialize([_asset(f'genius/{embed_table}')])

    assert res.success
    assert calls['first_arg'] is client
    assert calls['delete_finished'] is True
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == jobs_table
    (filter_fn,) = source.filters()
    assert filter_fn({'state': 'JOB_STATE_SUCCEEDED'})
    assert not filter_fn({'state': 'JOB_STATE_RUNNING'})

    (created,) = fake_pipelines
    (extract,) = created.pipeline.extracts
    assert extract.source is source
    assert extract.kwargs == {
        'table_name': embed_table,
        'loader_file_format': 'parquet',
    }
    assert created.pipeline.normalize_calls == 1
    assert loaded == [(created.pipeline, embed_table, True)]


@pytest.mark.parametrize('task, embed_table, jobs_table, lyrics_table', EMBED_CASES)
def test_embed_submit_asset_submits_new_jobs(
    task,
    embed_table,
    jobs_table,
    lyrics_table,
    monkeypatch,
    fake_sql_table,
    fake_pipelines,
):
    client = MagicMock()
    monkeypatch.setattr(embeddings, '_make_client', lambda: client)
    calls = {}

    def fake_submit_batch_file_job(texts, client=None, task=None, disp_name=None):
        calls['first_arg'] = texts
        calls['task'] = task
        calls['disp_name'] = disp_name
        return FakeResource('submit_batch_file_job')

    monkeypatch.setattr(embeddings, 'submit_batch_file_job', fake_submit_batch_file_job)

    res = dg.materialize([_asset(f'genius/{embed_table}_jobs_submit')])

    assert res.success
    assert calls['first_arg'] is client
    assert calls['task'] == task
    assert calls['disp_name'] == f'lyrics-batch-embed-file:{task}'
    (source,) = fake_sql_table
    assert source.call.kwargs['table'] == lyrics_table
    assert source.limits() == [N_EMBED_JOBS]

    (created,) = fake_pipelines
    (run,) = created.pipeline.runs
    assert run.kwargs == {
        'table_name': jobs_table,
        'write_disposition': 'append',
        'primary_key': 'name',
    }
